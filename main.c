/*
 * Sentinel  —  a fully self-contained Small Language Model in standard C.
 *
 * No external machine-learning libraries.  Everything below — the tokenizer,
 * the embedding table, the stacked recurrent layers, the matrix multiply, the
 * backprop-through-time trainer, the checkpoint format and the process-spawning
 * sub-agent layer — is implemented by hand using only the C standard library
 * (plus libm for tanh/exp/sqrt/log).
 *
 * Architecture (a real *deep* network, not a single shallow RNN):
 *
 *     token --> [embedding] --> [RNN layer 0] --> [RNN layer 1] --> ... --> [softmax] --> next char
 *                                    ^  |              ^  |
 *                                    |  v hidden state |  v hidden state  (fed back each step)
 *                                    +--+              +--+
 *
 * Because each layer keeps a hidden-state vector that is fed back on every
 * character, the model processes an arbitrarily long stream with no fixed
 * "token context window".
 *
 * Modes:
 *   ./sentinel                 train/continue training, print loss + a sample,
 *                               then read stdin: plain text -> online learning,
 *                               lines containing "TASK:" -> spawn a sub-agent.
 *   ./sentinel --agent "<t>"   run as a spawned sub-agent for task <t>.
 *   ./sentinel --train <file>  train extra epochs on an external corpus file.
 *
 * Build:  gcc -O2 -Wall -o sentinel main.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <strings.h>
#include <limits.h>

/* ------------------------------------------------------------------ */
/*  Hyperparameters — raise these to grow the network's capacity.      */
/*  The code below is dimension-generic, so changing them and          */
/*  recompiling is all that's needed to deepen / widen the model.      */
/* ------------------------------------------------------------------ */
#define VOCAB       128     /* character-level tokenizer: 7-bit ASCII   */
#define EMBED       64      /* learned embedding dimension              */
#define HIDDEN      192     /* hidden units per recurrent layer         */
#define NUM_LAYERS  3       /* stacked recurrent layers (depth)         */
#define SEQ_LEN     32      /* BPTT unroll length                       */
#define LR          0.10    /* base learning rate (Adagrad)             */
#define CLIP        5.0     /* gradient clipping bound                  */

#define CKPT_MAGIC  0x534C4D31u   /* "SLM1" */
#define CKPT_PATH   "sentinel.bin"
#define VERSION     "0.1.0"

/* ------------------------------------------------------------------ */
/*  Parameters (static weight matrices) + Adagrad memory + gradients.  */
/*  EMBED <= HIDDEN, so layer-0 input is zero-padded up to HIDDEN and   */
/*  every weight matrix can share a uniform HIDDEN-wide stride.         */
/* ------------------------------------------------------------------ */
static double Wemb[VOCAB][EMBED];                 /* embedding table      */
static double Wxh [NUM_LAYERS][HIDDEN][HIDDEN];   /* input  -> hidden     */
static double Whh [NUM_LAYERS][HIDDEN][HIDDEN];   /* hidden -> hidden     */
static double bh  [NUM_LAYERS][HIDDEN];           /* hidden bias          */
static double Why [VOCAB][HIDDEN];                 /* hidden -> output     */
static double by  [VOCAB];                         /* output bias          */

static double dWemb[VOCAB][EMBED];
static double dWxh [NUM_LAYERS][HIDDEN][HIDDEN];
static double dWhh [NUM_LAYERS][HIDDEN][HIDDEN];
static double dbh  [NUM_LAYERS][HIDDEN];
static double dWhy [VOCAB][HIDDEN];
static double dby  [VOCAB];

static double mWemb[VOCAB][EMBED];
static double mWxh [NUM_LAYERS][HIDDEN][HIDDEN];
static double mWhh [NUM_LAYERS][HIDDEN][HIDDEN];
static double mbh  [NUM_LAYERS][HIDDEN];
static double mWhy [VOCAB][HIDDEN];
static double mby  [VOCAB];

/* Per-timestep activation cache used by backprop-through-time. */
static int    g_in [SEQ_LEN];
static double g_emb[SEQ_LEN][EMBED];
static double g_h  [NUM_LAYERS][SEQ_LEN + 1][HIDDEN];  /* [l][0] = carried state */
static double g_y  [SEQ_LEN][VOCAB];
static double g_p  [SEQ_LEN][VOCAB];

/* ------------------------------------------------------------------ */
/*  Deterministic PRNG so the "static" default weights are reproducible.*/
/* ------------------------------------------------------------------ */
static unsigned long long rng_state = 0x9E3779B97F4A7C15ULL;
static double rnd_uniform(double lo, double hi) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    double u = (double)(rng_state >> 11) / (double)(1ULL << 53);
    return lo + u * (hi - lo);
}

/* ------------------------------------------------------------------ */
/*  Manual matrix * vector:  out[i] = sum_j M[i][j] * v[j].            */
/*  Hand written — this is the core linear-algebra primitive, used for */
/*  every layer at every timestep.  No BLAS, no libraries.             */
/* ------------------------------------------------------------------ */
static void matvec(double *out, const double *M, const double *v,
                   int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        const double *row = M + (size_t)i * cols;
        double acc = 0.0;
        for (int j = 0; j < cols; j++)
            acc += row[j] * v[j];
        out[i] = acc;
    }
}

/* Numerically stable softmax over n logits. */
static void softmax(double *p, const double *x, int n) {
    double mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { p[i] = exp(x[i] - mx); sum += p[i]; }
    double inv = 1.0 / sum;
    for (int i = 0; i < n; i++) p[i] *= inv;
}

/* ------------------------------------------------------------------ */
/*  Weight initialisation (fixed seed -> reproducible static weights). */
/* ------------------------------------------------------------------ */
static void init_weights(void) {
    rng_state = 0x9E3779B97F4A7C15ULL;
    for (int t = 0; t < VOCAB; t++)
        for (int j = 0; j < EMBED; j++)
            Wemb[t][j] = rnd_uniform(-0.1, 0.1);
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int i = 0; i < HIDDEN; i++) {
            for (int j = 0; j < HIDDEN; j++) {
                Wxh[l][i][j] = rnd_uniform(-0.1, 0.1);
                Whh[l][i][j] = rnd_uniform(-0.1, 0.1);
            }
            bh[l][i] = 0.0;
        }
    }
    for (int i = 0; i < VOCAB; i++) {
        for (int j = 0; j < HIDDEN; j++) Why[i][j] = rnd_uniform(-0.1, 0.1);
        by[i] = 0.0;
    }
    memset(mWemb, 0, sizeof mWemb); memset(mWxh, 0, sizeof mWxh);
    memset(mWhh, 0, sizeof mWhh);   memset(mbh, 0, sizeof mbh);
    memset(mWhy, 0, sizeof mWhy);   memset(mby, 0, sizeof mby);
}

/* ------------------------------------------------------------------ */
/*  One BPTT step over a SEQ_LEN window.  Returns the cross-entropy    */
/*  loss; updates weights in place (Adagrad) and carries hidden state. */
/* ------------------------------------------------------------------ */
static void adagrad(double *p, double *g, double *m, int n) {
    for (int i = 0; i < n; i++) {
        double grad = g[i];
        if (grad >  CLIP) grad =  CLIP;
        if (grad < -CLIP) grad = -CLIP;
        m[i] += grad * grad;
        p[i] -= LR * grad / sqrt(m[i] + 1e-8);
    }
}

static double train_step(const int *inputs, const int *targets,
                         double hprev[NUM_LAYERS][HIDDEN]) {
    /* ---- forward ---- */
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int i = 0; i < HIDDEN; i++) g_h[l][0][i] = hprev[l][i];

    double loss = 0.0;
    double a[HIDDEN], b[HIDDEN], xin[HIDDEN];
    for (int t = 0; t < SEQ_LEN; t++) {
        int tok = inputs[t];
        g_in[t] = tok;
        for (int j = 0; j < EMBED; j++) g_emb[t][j] = Wemb[tok][j];

        for (int l = 0; l < NUM_LAYERS; l++) {
            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = g_emb[t][j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = g_h[l - 1][t + 1][j];
            }
            matvec(a, &Wxh[l][0][0], xin,            HIDDEN, HIDDEN);
            matvec(b, &Whh[l][0][0], g_h[l][t],       HIDDEN, HIDDEN);
            for (int i = 0; i < HIDDEN; i++)
                g_h[l][t + 1][i] = tanh(a[i] + b[i] + bh[l][i]);
        }
        matvec(g_y[t], &Why[0][0], g_h[NUM_LAYERS - 1][t + 1], VOCAB, HIDDEN);
        for (int i = 0; i < VOCAB; i++) g_y[t][i] += by[i];
        softmax(g_p[t], g_y[t], VOCAB);
        loss += -log(g_p[t][targets[t]] + 1e-12);
    }

    /* ---- backward (through depth and time) ---- */
    memset(dWemb, 0, sizeof dWemb); memset(dWxh, 0, sizeof dWxh);
    memset(dWhh, 0, sizeof dWhh);   memset(dbh, 0, sizeof dbh);
    memset(dWhy, 0, sizeof dWhy);   memset(dby, 0, sizeof dby);

    double dh_next[NUM_LAYERS][HIDDEN];
    memset(dh_next, 0, sizeof dh_next);

    for (int t = SEQ_LEN - 1; t >= 0; t--) {
        double dy[VOCAB];
        for (int i = 0; i < VOCAB; i++) dy[i] = g_p[t][i];
        dy[targets[t]] -= 1.0;

        const double *top = g_h[NUM_LAYERS - 1][t + 1];
        for (int i = 0; i < VOCAB; i++) {
            dby[i] += dy[i];
            for (int j = 0; j < HIDDEN; j++) dWhy[i][j] += dy[i] * top[j];
        }

        double dhtot[NUM_LAYERS][HIDDEN];
        for (int l = 0; l < NUM_LAYERS; l++)
            for (int i = 0; i < HIDDEN; i++) dhtot[l][i] = dh_next[l][i];
        for (int j = 0; j < HIDDEN; j++) {
            double s = 0.0;
            for (int i = 0; i < VOCAB; i++) s += Why[i][j] * dy[i];
            dhtot[NUM_LAYERS - 1][j] += s;
        }

        for (int l = NUM_LAYERS - 1; l >= 0; l--) {
            double dtanh[HIDDEN];
            for (int i = 0; i < HIDDEN; i++) {
                double h = g_h[l][t + 1][i];
                dtanh[i] = dhtot[l][i] * (1.0 - h * h);
                dbh[l][i] += dtanh[i];
            }
            for (int i = 0; i < HIDDEN; i++) {
                double dti = dtanh[i];
                for (int k = 0; k < HIDDEN; k++)
                    dWhh[l][i][k] += dti * g_h[l][t][k];
            }
            /* gradient to previous time, same layer (temporal) */
            for (int k = 0; k < HIDDEN; k++) {
                double s = 0.0;
                for (int i = 0; i < HIDDEN; i++) s += Whh[l][i][k] * dtanh[i];
                dh_next[l][k] = s;
            }
            /* layer input (xin) for this t */
            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = g_emb[t][j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = g_h[l - 1][t + 1][j];
            }
            for (int i = 0; i < HIDDEN; i++) {
                double dti = dtanh[i];
                for (int j = 0; j < HIDDEN; j++)
                    dWxh[l][i][j] += dti * xin[j];
            }
            /* gradient to the layer input */
            double dxin[HIDDEN];
            for (int j = 0; j < HIDDEN; j++) {
                double s = 0.0;
                for (int i = 0; i < HIDDEN; i++) s += Wxh[l][i][j] * dtanh[i];
                dxin[j] = s;
            }
            if (l == 0) {
                int tok = g_in[t];
                for (int j = 0; j < EMBED; j++) dWemb[tok][j] += dxin[j];
            } else {
                for (int j = 0; j < HIDDEN; j++) dhtot[l - 1][j] += dxin[j];
            }
        }
    }

    /* ---- Adagrad update ---- */
    adagrad(&Wemb[0][0], &dWemb[0][0], &mWemb[0][0], VOCAB * EMBED);
    adagrad(&Why[0][0],  &dWhy[0][0],  &mWhy[0][0],  VOCAB * HIDDEN);
    adagrad(by, dby, mby, VOCAB);
    for (int l = 0; l < NUM_LAYERS; l++) {
        adagrad(&Wxh[l][0][0], &dWxh[l][0][0], &mWxh[l][0][0], HIDDEN * HIDDEN);
        adagrad(&Whh[l][0][0], &dWhh[l][0][0], &mWhh[l][0][0], HIDDEN * HIDDEN);
        adagrad(bh[l], dbh[l], mbh[l], HIDDEN);
    }

    /* carry the last hidden state forward (continuous, no context limit) */
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int i = 0; i < HIDDEN; i++) hprev[l][i] = g_h[l][SEQ_LEN][i];
    return loss;
}

/* ------------------------------------------------------------------ */
/*  Train on a text buffer for a number of iterations.                 */
/* ------------------------------------------------------------------ */
static void train(const char *data, int data_len, int iters, int verbose) {
    if (data_len < SEQ_LEN + 1) return;
    double hprev[NUM_LAYERS][HIDDEN];
    memset(hprev, 0, sizeof hprev);
    int p = 0;
    double smooth = -1.0;
    int inputs[SEQ_LEN], targets[SEQ_LEN];

    for (int it = 0; it < iters; it++) {
        if (p + SEQ_LEN + 1 >= data_len) { p = 0; memset(hprev, 0, sizeof hprev); }
        for (int k = 0; k < SEQ_LEN; k++) {
            inputs[k]  = (unsigned char)data[p + k]     & (VOCAB - 1);
            targets[k] = (unsigned char)data[p + k + 1] & (VOCAB - 1);
        }
        double loss = train_step(inputs, targets, hprev);
        double per_char = loss / SEQ_LEN;
        smooth = (smooth < 0.0) ? per_char : 0.999 * smooth + 0.001 * per_char;
        p += SEQ_LEN;
        if (verbose && (it % 200 == 0 || it == iters - 1))
            printf("  iter %5d/%d   loss/char = %.4f\n", it, iters, smooth);
    }
}

/* ------------------------------------------------------------------ */
/*  Generate text by sampling the model character by character.        */
/* ------------------------------------------------------------------ */
static void sample(int seed, int n) {
    double h[NUM_LAYERS][HIDDEN];
    memset(h, 0, sizeof h);
    double a[HIDDEN], b[HIDDEN], xin[HIDDEN], emb[EMBED], y[VOCAB], p[VOCAB];
    int tok = seed & (VOCAB - 1);

    for (int step = 0; step < n; step++) {
        for (int j = 0; j < EMBED; j++) emb[j] = Wemb[tok][j];
        for (int l = 0; l < NUM_LAYERS; l++) {
            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = emb[j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = h[l - 1][j];
            }
            matvec(a, &Wxh[l][0][0], xin,  HIDDEN, HIDDEN);
            matvec(b, &Whh[l][0][0], h[l], HIDDEN, HIDDEN);
            for (int i = 0; i < HIDDEN; i++)
                h[l][i] = tanh(a[i] + b[i] + bh[l][i]);
        }
        matvec(y, &Why[0][0], h[NUM_LAYERS - 1], VOCAB, HIDDEN);
        for (int i = 0; i < VOCAB; i++) y[i] += by[i];
        softmax(p, y, VOCAB);

        double r = (double)rand() / ((double)RAND_MAX + 1.0), c = 0.0;
        int next = VOCAB - 1;
        for (int i = 0; i < VOCAB; i++) { c += p[i]; if (r < c) { next = i; break; } }
        int ch = next;
        putchar(isprint(ch) || ch == '\n' ? ch : ' ');
        tok = next;
    }
    putchar('\n');
}

/* ------------------------------------------------------------------ */
/*  Checkpoint persistence: lets the model accumulate skills across     */
/*  runs.  The architecture dims are written in a header; on a mismatch  */
/*  we fall back to a fresh initialisation rather than load garbage.    */
/* ------------------------------------------------------------------ */
static void save_model(const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    unsigned hdr[6] = { CKPT_MAGIC, VOCAB, EMBED, HIDDEN, NUM_LAYERS, 0 };
    fwrite(hdr, sizeof(unsigned), 6, f);
    fwrite(Wemb, sizeof Wemb, 1, f);
    fwrite(Wxh,  sizeof Wxh,  1, f);
    fwrite(Whh,  sizeof Whh,  1, f);
    fwrite(bh,   sizeof bh,   1, f);
    fwrite(Why,  sizeof Why,  1, f);
    fwrite(by,   sizeof by,   1, f);
    fwrite(mWemb, sizeof mWemb, 1, f);
    fwrite(mWxh,  sizeof mWxh,  1, f);
    fwrite(mWhh,  sizeof mWhh,  1, f);
    fwrite(mbh,   sizeof mbh,   1, f);
    fwrite(mWhy,  sizeof mWhy,  1, f);
    fwrite(mby,   sizeof mby,   1, f);
    fclose(f);
}

static int load_model(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned hdr[6];
    if (fread(hdr, sizeof(unsigned), 6, f) != 6 ||
        hdr[0] != CKPT_MAGIC || hdr[1] != VOCAB || hdr[2] != EMBED ||
        hdr[3] != HIDDEN || hdr[4] != NUM_LAYERS) {
        fclose(f);
        return 0;   /* architecture changed -> start fresh, safely */
    }
    int ok = 1;
    ok &= fread(Wemb, sizeof Wemb, 1, f) == 1;
    ok &= fread(Wxh,  sizeof Wxh,  1, f) == 1;
    ok &= fread(Whh,  sizeof Whh,  1, f) == 1;
    ok &= fread(bh,   sizeof bh,   1, f) == 1;
    ok &= fread(Why,  sizeof Why,  1, f) == 1;
    ok &= fread(by,   sizeof by,   1, f) == 1;
    ok &= fread(mWemb, sizeof mWemb, 1, f) == 1;
    ok &= fread(mWxh,  sizeof mWxh,  1, f) == 1;
    ok &= fread(mWhh,  sizeof mWhh,  1, f) == 1;
    ok &= fread(mbh,   sizeof mbh,   1, f) == 1;
    ok &= fread(mWhy,  sizeof mWhy,  1, f) == 1;
    ok &= fread(mby,   sizeof mby,   1, f) == 1;
    fclose(f);
    return ok;
}

/* ================================================================== */
/*  Sub-agent layer                                                    */
/* ================================================================== */

/* Resolve an absolute path to this very executable for re-exec. */
static void self_path(char *buf, size_t n, const char *argv0) {
    ssize_t r = readlink("/proc/self/exe", buf, n - 1);
    if (r > 0) { buf[r] = '\0'; return; }
    snprintf(buf, n, "%s", argv0);
}

/*
 * Hard safety guard.  Returns 0 (unsafe) if the command string contains
 * sudo or any destructive / privileged pattern.  `sudo` is non-negotiable.
 */
static int is_command_safe(const char *cmd) {
    char low[1024];
    size_t n = strlen(cmd);
    if (n >= sizeof low) n = sizeof low - 1;
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)cmd[i]);
    low[n] = '\0';

    static const char *deny[] = {
        "sudo", "su -", "doas",
        "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -fr /",
        "mkfs", "dd if=", "dd of=", "of=/dev/", "> /dev/sd", ">/dev/sd",
        ":(){", "fork()", "shutdown", "reboot", "halt", "init 0", "init 6",
        "chmod -r 777 /", "chmod 777 /", "chown -r", "> /dev/sda",
        "mv / ", "wget http", "curl http",       /* block remote fetch+run */
        "|sh", "| sh", "|bash", "| bash",
        "/etc/passwd", "/etc/shadow", "crontab", "iptables -f",
        NULL
    };
    for (int i = 0; deny[i]; i++)
        if (strstr(low, deny[i])) return 0;
    return 1;
}

/* Light keyword classifier: 0 = cyber, 1 = coding, 2 = general. */
static int classify_task(const char *task) {
    char low[512];
    size_t n = strlen(task);
    if (n >= sizeof low) n = sizeof low - 1;
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)task[i]);
    low[n] = '\0';

    static const char *cyber[] = {
        "scan","port","nmap","vuln","exploit","sql","inject","xss","recon",
        "firewall","pentest","hash","encrypt","decrypt","payload","cve",
        "security","malware","forensic","network","packet","ssl","tls", NULL };
    static const char *code[] = {
        "write","function","code","compile","program","script","reverse",
        "algorithm","python","java","rust","golang"," c ","header","struct",
        "loop","array","string","sort","build","refactor","debug", NULL };

    for (int i = 0; cyber[i]; i++) if (strstr(low, cyber[i])) return 0;
    for (int i = 0; code[i];  i++) if (strstr(low, code[i]))  return 1;
    return 2;
}

/*
 * Build a concrete, safe shell command for a task.  `cmd` is filled in.
 * Returns the classification used.  For coding tasks it may first emit a
 * small C snippet to /tmp via plain C file I/O, then compile + run it.
 */
/*
 * Pull safe search keywords out of a task string: alphanumeric tokens of
 * length >= 4 that aren't common stop-words, joined by '|' into a grep -E
 * alternation. Only [A-Za-z0-9|] ever reaches the shell, so this can't be
 * used for command injection.
 */
static void extract_keywords(const char *task, char *out, size_t outsz) {
    static const char *stop[] = {
        "show","find","that","this","with","from","what","which","your","you",
        "might","right","there","here","about","please","could","would","some",
        "look","give","tell","need","want","have","does","into","them","they",
        "cve","cves","vuln","vulnerability","vulnerabilities", NULL };
    char tok[64];
    size_t oi = 0; int first = 1;
    const char *p = task;
    while (*p && oi + 2 < outsz) {
        while (*p && !isalnum((unsigned char)*p)) p++;
        size_t ti = 0;
        while (*p && isalnum((unsigned char)*p) && ti < sizeof tok - 1)
            tok[ti++] = (char)tolower((unsigned char)*p++);
        tok[ti] = '\0';
        if (ti < 4) continue;
        int skip = 0;
        for (int i = 0; stop[i]; i++) if (strcmp(tok, stop[i]) == 0) { skip = 1; break; }
        if (skip) continue;
        if (!first && oi + 1 < outsz) out[oi++] = '|';
        for (size_t i = 0; i < ti && oi + 1 < outsz; i++) out[oi++] = tok[i];
        first = 0;
    }
    out[oi] = '\0';
    if (oi == 0)   /* nothing useful -> default to mobile/common terms */
        snprintf(out, outsz, "android|ios|mobile|bluetooth|wifi|kernel|webkit");
}

static int build_command(const char *task, char *cmd, size_t cmdsz) {
    int kind = classify_task(task);
    char low[512];
    size_t n = strlen(task);
    if (n >= sizeof low) n = sizeof low - 1;
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)task[i]);
    low[n] = '\0';

    /* CVE / vulnerability lookup: search the fetched corpus and show real hits. */
    if (strstr(low, "cve") || strstr(low, "vuln")) {
        char kw[256];
        extract_keywords(task, kw, sizeof kw);
        snprintf(cmd, cmdsz,
            "dir=corpus; [ -d \"$dir\" ] || dir=.; "
            "echo '[cve agent] searching '\"$dir\"' for: %s'; "
            "hits=$(grep -rIl -i -E '%s' \"$dir\" --include='*.md' 2>/dev/null | head -3); "
            "if [ -z \"$hits\" ]; then "
            "echo 'no keyword match — showing a sample real CVE from the corpus:'; "
            "hits=$(grep -rIl -E 'CVE-[0-9]{4}-[0-9]+' \"$dir\" --include='*.md' 2>/dev/null | head -1); fi; "
            "if [ -z \"$hits\" ]; then "
            "echo 'no CVE writeups in corpus — run ./fetch_corpus.sh (recent years) first'; "
            "else for f in $hits; do echo \"=== $f ===\"; "
            "grep -i -m1 -E 'CVE-[0-9]{4}-[0-9]+' \"$f\"; "
            "sed -n '/### Description/,/###/p' \"$f\" | sed '1d;$d' | head -8; echo; done; fi",
            kw, kw);
        return 0;  /* report as cyber specialism */
    }

    if (strstr(low, "list") && (strstr(low, "file") || strstr(low, "dir"))) {
        snprintf(cmd, cmdsz, "ls -la");
        return kind;
    }

    if (kind == 0) { /* cyber: read-only local recon only */
        if (strstr(low, "port") || strstr(low, "listen") || strstr(low, "network"))
            snprintf(cmd, cmdsz,
                "echo '[recon] local listening sockets:'; "
                "(ss -tlnp 2>/dev/null || netstat -tlnp 2>/dev/null || "
                "echo 'no socket tool available') | head -n 15");
        else if (strstr(low, "hash") || strstr(low, "sha") || strstr(low, "integrity"))
            snprintf(cmd, cmdsz,
                "echo -n 'hash-this-sample-input' | sha256sum");
        else if (strstr(low, "nmap") || strstr(low, "scan"))
            snprintf(cmd, cmdsz,
                "command -v nmap >/dev/null && echo 'nmap is available' "
                "|| echo 'nmap not installed (recon stub) — would scan authorised targets only'");
        else
            snprintf(cmd, cmdsz,
                "echo '[cyber agent] environment fingerprint:'; uname -a; id -un");
        return kind;
    }

    if (kind == 1) { /* coding: actually write, compile and run a C snippet */
        const char *src = "/tmp/slm_agent_snippet.c";
        const char *bin = "/tmp/slm_agent_snippet";
        FILE *fp = fopen(src, "w");
        if (fp) {
            if (strstr(low, "reverse") && strstr(low, "string")) {
                fputs(
                    "#include <stdio.h>\n#include <string.h>\n"
                    "void rev(char*s){int i=0,j=strlen(s)-1;while(i<j){char t=s[i];s[i]=s[j];s[j]=t;i++;j--;}}\n"
                    "int main(){char b[]=\"cybersecurity\";rev(b);printf(\"reversed: %s\\n\",b);return 0;}\n",
                    fp);
            } else {
                fputs(
                    "#include <stdio.h>\n"
                    "int main(){printf(\"[coding agent] hello from a compiled C sub-agent\\n\");"
                    "for(int i=1;i<=5;i++)printf(\"  fib-ish %d\\n\",i*i);return 0;}\n",
                    fp);
            }
            fclose(fp);
            snprintf(cmd, cmdsz,
                "gcc %s -o %s 2>&1 && echo '[coding agent] compiled OK, running:' && %s",
                src, bin, bin);
        } else {
            snprintf(cmd, cmdsz, "echo '[coding agent] could not write snippet'");
        }
        return kind;
    }

    /* general */
    snprintf(cmd, cmdsz,
        "echo '[general agent] working dir + date:'; pwd; date");
    return kind;
}

/* Runs as the spawned sub-agent. */
static int run_agent(const char *task) {
    const char *kinds[] = { "CYBERSECURITY", "CODING", "GENERAL" };
    char cmd[2048];
    int kind = build_command(task, cmd, sizeof cmd);

    printf("\n  ┌─ sub-agent pid=%d  specialism=%s\n", (int)getpid(), kinds[kind]);
    printf("  │  task: %s\n", task);

    if (getenv("SLM_NO_EXEC")) {
        printf("  │  [plan-only mode: SLM_NO_EXEC set, not executing]\n");
        printf("  │  would run: %s\n", cmd);
        printf("  └─ done (pid=%d)\n", (int)getpid());
        return 0;
    }

    if (!is_command_safe(task) || !is_command_safe(cmd)) {
        printf("  │  REFUSED by safety guard: command contains a blocked "
               "(sudo/destructive) pattern.\n");
        printf("  └─ no action taken (pid=%d)\n", (int)getpid());
        return 2;
    }

    printf("  │  exec: %s\n  │  ----- output -----\n", cmd);
    fflush(stdout);

    FILE *pp = popen(cmd, "r");
    if (!pp) {
        printf("  │  failed to launch shell\n  └─ error (pid=%d)\n", (int)getpid());
        return 1;
    }
    char line[512];
    while (fgets(line, sizeof line, pp)) printf("  │  %s", line);
    int rc = pclose(pp);
    printf("\n  └─ sub-agent pid=%d finished (exit code %d)\n",
           (int)getpid(), rc == -1 ? -1 : WEXITSTATUS(rc));
    return 0;
}

/* Parent forks + execs a fresh sub-agent process for `task`. */
static void spawn_agent(const char *selfexe, const char *task) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return; }
    if (pid == 0) {
        execl(selfexe, selfexe, "--agent", task, (char *)NULL);
        perror("execl");
        _exit(127);
    }
    int status = 0;
    waitpid(pid, &status, 0);
}

/* ================================================================== */
/*  Data consolidation                                                 */
/*                                                                     */
/*  More parameters need more data.  These helpers fold many text /    */
/*  code files (or a whole directory tree) into ONE training corpus    */
/*  held in a growable buffer, so a bigger network has enough to learn */
/*  from.  Used by `--train <path...>`.                                */
/* ================================================================== */
#define CORPUS_CAP_MAX (64u * 1024u * 1024u)   /* 64 MB safety ceiling */

typedef struct { char *buf; size_t len; size_t cap; int files; } Corpus;

static void corpus_reserve(Corpus *c, size_t extra) {
    if (c->len + extra + 1 <= c->cap) return;
    size_t want = c->cap ? c->cap : 65536;
    while (want < c->len + extra + 1) want *= 2;
    if (want > CORPUS_CAP_MAX) want = CORPUS_CAP_MAX;
    char *n = realloc(c->buf, want);
    if (n) { c->buf = n; c->cap = want; }
}

/* Is this a text-ish file worth feeding to a character model? */
static int has_text_ext(const char *name) {
    static const char *ext[] = {
        ".txt",".md",".c",".h",".cpp",".hpp",".cc",".py",".js",".ts",
        ".json",".yaml",".yml",".cfg",".ini",".log",".csv",".html",".css",
        ".java",".rs",".go",".sh",".rb",".php",".sql",".xml",".tex", NULL };
    const char *dot = strrchr(name, '.');
    if (!dot) return 0;
    for (int i = 0; ext[i]; i++) if (strcasecmp(dot, ext[i]) == 0) return 1;
    return 0;
}

static void corpus_append_file(Corpus *c, const char *path) {
    if (c->len >= CORPUS_CAP_MAX) return;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    char chunk[65536];
    size_t r, added = 0;
    while ((r = fread(chunk, 1, sizeof chunk, f)) > 0) {
        if (c->len >= CORPUS_CAP_MAX) break;
        corpus_reserve(c, r);
        if (c->len + r > c->cap - 1) r = (c->cap > c->len + 1) ? c->cap - 1 - c->len : 0;
        if (r == 0) break;
        memcpy(c->buf + c->len, chunk, r);
        c->len += r; added += r;
    }
    fclose(f);
    if (added) {
        corpus_reserve(c, 2);
        if (c->len + 1 < c->cap) c->buf[c->len++] = '\n';
        c->files++;
        printf("  + consolidated %-48s (%zu bytes)\n", path, added);
    }
}

/* Recursively walk a directory tree, consolidating every text file. */
static void corpus_append_dir(Corpus *c, const char *dir, int depth) {
    if (depth > 16 || c->len >= CORPUS_CAP_MAX) return;
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    char path[PATH_MAX];
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] == '.') continue;          /* skip . .. and dotfiles (.git) */
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
        struct stat st;
        if (stat(path, &st) != 0) continue;
        if (S_ISDIR(st.st_mode))      corpus_append_dir(c, path, depth + 1);
        else if (S_ISREG(st.st_mode) && has_text_ext(e->d_name))
                                       corpus_append_file(c, path);
    }
    closedir(d);
}

/* Consolidate a path that may be a file or a directory. */
static void corpus_append_path(Corpus *c, const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) { fprintf(stderr, "  ! skip %s (not found)\n", path); return; }
    if (S_ISDIR(st.st_mode)) corpus_append_dir(c, path, 0);
    else                     corpus_append_file(c, path);
}

/* ================================================================== */
/*  Embedded bootstrap corpus (cybersecurity + coding flavoured).      */
/* ================================================================== */
static const char CORPUS[] =
    "the local slm studies cybersecurity and coding. a secure system "
    "validates every input, never trusts user data, and patches known "
    "vulnerabilities quickly. scan the network, enumerate open ports, "
    "and check for sql injection and cross site scripting. write clean "
    "c code: declare functions, allocate memory, free what you allocate, "
    "and handle every error. hash passwords, encrypt secrets, and rotate "
    "keys. a firewall blocks untrusted traffic while logging suspicious "
    "packets. compile the program, run the tests, fix the bugs, repeat. "
    "defense in depth means many layers: authentication, authorization, "
    "encryption, monitoring, and least privilege for every process. ";

/* ================================================================== */
/*  main                                                               */
/* ================================================================== */
static void print_usage(const char *prog) {
    printf(
"Sentinel v%s — a self-contained deep-RNN AI agent in pure C.\n"
"A character-level neural net (no libraries) that learns from text, answers\n"
"CVE questions from its corpus, and forks sub-agents to do small tasks.\n"
"\n"
"USAGE\n"
"  %s                      train on the built-in/checkpointed model, print a\n"
"                            sample, then enter the interactive AI loop\n"
"  %s --train <path...>    consolidate files/dirs into a corpus and train on it\n"
"  %s --agent \"<task>\"     run once as a single sub-agent for <task>\n"
"  %s --help | -h          show this help\n"
"  %s --version            print version\n"
"\n"
"INTERACTIVE LOOP (what to type once it's running)\n"
"  TASK: <something>         spawn a sub-agent (cyber / coding / general):\n"
"      TASK: list files in this directory\n"
"      TASK: write a C function to reverse a string\n"
"      TASK: scan local network ports\n"
"  show me a cve about <x>   ask about a vulnerability — searches your corpus\n"
"                            and prints a real CVE (no 'TASK:' needed)\n"
"  <any other text>          is absorbed as live training data (online learning)\n"
"  Ctrl-D                    quit (the model is checkpointed on exit)\n"
"\n"
"USING IT AS AN AI (typical first run)\n"
"  ./start.sh                build + fetch data + train + launch, one command\n"
"  ./fetch_corpus.sh         download real security/coding/CVE data into ./corpus\n"
"  %s --train corpus       teach it everything you fetched\n"
"\n"
"ENVIRONMENT VARIABLES\n"
"  SLM_EPOCHS=N              training iterations (default 2000)\n"
"  SLM_NO_EXEC=1             agents PLAN ONLY — print commands, run nothing\n"
"\n"
"SAFETY\n"
"  Sub-agents run real shell commands but a hard guard refuses 'sudo' and\n"
"  destructive patterns (rm -rf /, mkfs, dd, fork bombs, shutdown, ...).\n"
"  Set SLM_NO_EXEC=1 to disable command execution entirely.\n",
        VERSION, prog, prog, prog, prog, prog, prog);
}

int main(int argc, char **argv) {
    /* ---- help / version ---- */
    if (argc >= 2 && (strcmp(argv[1], "--help") == 0 ||
                      strcmp(argv[1], "-h") == 0 ||
                      strcmp(argv[1], "help") == 0)) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc >= 2 && (strcmp(argv[1], "--version") == 0 ||
                      strcmp(argv[1], "-v") == 0)) {
        printf("Sentinel v%s\n", VERSION);
        return 0;
    }

    /* ---- sub-agent mode ---- */
    if (argc >= 3 && strcmp(argv[1], "--agent") == 0)
        return run_agent(argv[2]);

    srand(1234567u);
    init_weights();

    int loaded = load_model(CKPT_PATH);
    if (loaded) printf("Loaded checkpoint '%s' — continuing to learn.\n", CKPT_PATH);
    else        printf("No checkpoint — starting from fixed static weights.\n");

    int iters = 2000;
    const char *env_ep = getenv("SLM_EPOCHS");
    if (env_ep) { int v = atoi(env_ep); if (v > 0) iters = v; }

    /* ---- external-corpus training mode (with data consolidation) ---- */
    if (argc >= 3 && strcmp(argv[1], "--train") == 0) {
        Corpus c = {0};
        printf("Consolidating training data from %d path(s):\n", argc - 2);
        for (int i = 2; i < argc; i++) corpus_append_path(&c, argv[i]);
        if (c.len < SEQ_LEN + 1) {
            fprintf(stderr, "consolidated corpus too small (%zu bytes)\n", c.len);
            free(c.buf);
            return 1;
        }
        c.buf[c.len] = '\0';
        printf("Consolidated %d file(s) -> %zu bytes. Training for %d iters...\n",
               c.files, c.len, iters);
        train(c.buf, (int)c.len, iters, 1);
        free(c.buf);
        save_model(CKPT_PATH);
        printf("Checkpoint saved to '%s'.\n", CKPT_PATH);
        return 0;
    }

    /* ---- default mode: train on the embedded corpus, sample, then serve ---- */
    printf("=== Sentinel : self-contained deep RNN ===\n");
    printf("arch: VOCAB=%d EMBED=%d HIDDEN=%d LAYERS=%d SEQ_LEN=%d  (%d params)\n",
           VOCAB, EMBED, HIDDEN, NUM_LAYERS, SEQ_LEN,
           (int)(VOCAB*EMBED + NUM_LAYERS*(2*HIDDEN*HIDDEN + HIDDEN) + VOCAB*HIDDEN + VOCAB));
    printf("Training loop (BPTT through %d stacked layers):\n", NUM_LAYERS);
    train(CORPUS, (int)(sizeof(CORPUS) - 1), iters, 1);

    printf("\nGenerated sample after training (seeded with 't'):\n  \"");
    sample('t', 180);
    printf("\"\n");

    save_model(CKPT_PATH);
    printf("Checkpoint saved to '%s'.\n", CKPT_PATH);

    /* ---- serve loop: read stdin, online-learn, spawn agents on TASK: ---- */
    char selfexe[PATH_MAX];
    self_path(selfexe, sizeof selfexe, argv[0]);

    printf("\n=== read loop ===\n");
    printf("Type text to teach the model online, or 'TASK: ...' to spawn a "
           "sub-agent. (EOF / Ctrl-D to stop)\n");
    fflush(stdout);

    char line[4096];
    int spawned = 0;
    double hprev[NUM_LAYERS][HIDDEN];
    memset(hprev, 0, sizeof hprev);

    while (fgets(line, sizeof line, stdin)) {
        size_t len = strlen(line);
        while (len && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;

        char *trig = strstr(line, "TASK:");
        /* also treat a plain CVE/vuln question as a lookup, no "TASK:" needed */
        char low[256]; size_t ln = len < sizeof low - 1 ? len : sizeof low - 1;
        for (size_t i = 0; i < ln; i++) low[i] = (char)tolower((unsigned char)line[i]);
        low[ln] = '\0';
        int is_cve_q = (!trig) && (strstr(low, "cve") || strstr(low, "vuln"));

        if (trig || is_cve_q) {
            char *task = trig ? trig + 5 : line;
            while (*task == ' ') task++;
            printf("[trigger detected] spawning sub-agent for: \"%s\"\n", task);
            fflush(stdout);
            spawn_agent(selfexe, task);
            spawned++;
        } else {
            /* online learning: turn the incoming stream into weight updates */
            if (len >= SEQ_LEN + 1) {
                int inp[SEQ_LEN], tgt[SEQ_LEN];
                for (int s = 0; s + SEQ_LEN + 1 <= (int)len; s += SEQ_LEN) {
                    for (int k = 0; k < SEQ_LEN; k++) {
                        inp[k] = (unsigned char)line[s + k]     & (VOCAB - 1);
                        tgt[k] = (unsigned char)line[s + k + 1] & (VOCAB - 1);
                    }
                    train_step(inp, tgt, hprev);
                }
                printf("[online-learn] absorbed %zu chars into the network.\n", len);
            } else {
                printf("[input too short to train on: \"%s\"]\n", line);
            }
        }
        fflush(stdout);
    }

    save_model(CKPT_PATH);
    printf("\n=== session over ===\n");
    printf("sub-agents spawned this session: %d\n", spawned);
    printf("checkpoint persisted to '%s' (skills retained for next run).\n", CKPT_PATH);
    return 0;
}
