/*
 * Sentinel  —  a fully self-contained Small Language Model in standard C.
 *
 * No external machine-learning libraries.  Everything below — the tokenizer,
 * the embedding table, the stacked GRU layers, the matrix multiply, the
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
#include <signal.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

/* ------------------------------------------------------------------ */
/*  Hyperparameters — raise these to grow the network's capacity.      */
/*  The code below is dimension-generic, so changing them and          */
/*  recompiling is all that's needed to deepen / widen the model.      */
/* ------------------------------------------------------------------ */
/* The size knobs are -D-overridable so you can build BOTH a big model and a
 * small SLM from the same source, e.g.:
 *   gcc -O2 -mcmodel=large -o sentinel     main.c -lm                 (big, default)
 *   gcc -O2 -DHIDDEN=256 -DEMBED=64 -DNUM_LAYERS=2 -o sentinel-slm main.c -lm  (small) */
#ifndef VOCAB
#define VOCAB       128     /* character-level tokenizer: 7-bit ASCII   */
#endif
#ifndef EMBED
#define EMBED       128     /* learned embedding dimension              */
#endif
#ifndef HIDDEN
#define HIDDEN      1792    /* hidden units per GRU layer               */
#endif
#ifndef NUM_LAYERS
#define NUM_LAYERS  3       /* stacked GRU layers (depth)               */
#endif
#ifndef SEQ_LEN
#define SEQ_LEN     32      /* BPTT unroll length                       */
#endif
#define LR          0.002   /* base learning rate (Adam)                */
#define WARMUP      200     /* linear LR warmup steps (0 -> LR) for a    */
                            /* fresh model; stabilises early training,   */
                            /* then holds constant so online learning    */
                            /* keeps working across runs.                */
#define CLIP        5.0     /* gradient clipping bound                  */
#define ADAM_B1     0.9     /* Adam first-moment decay                  */
#define ADAM_B2     0.999   /* Adam second-moment decay                 */
#define ADAM_EPS    1e-8    /* Adam epsilon                             */
/* ~58M parameters and ~1.85 GB RAM at these defaults — this is the MAX that
 * still fits under the 2 GB static-data limit, so it builds with plain gcc
 * (no -mcmodel=large). Comfortable on a 4 GB+ machine; tight on exactly 2 GB.
 * With Adam each weight carries two moment estimates, so memory is ~32
 * bytes/param (weight + gradient + m + v, all double).
 *   Past ~58M params you cross 2 GB and must add -mcmodel=large (see 'make
 *   huge' for the 101M model). Lower HIDDEN if you want more RAM headroom.
 * RAM is not the wall; CPU is: training is O(NUM_LAYERS * HIDDEN^2 * SEQ_LEN)
 * per step (~4 billion mults/step here), so a real train at this size takes
 * hours on a Pi — train on a faster box and copy sentinel.bin, or run
 * overnight. Lower HIDDEN for faster iteration. To push past ~80M params,
 * switch double->float (halves RAM, speeds matmuls). VOCAB must stay a power
 * of two (the tokenizer masks with VOCAB-1). Dimension-generic otherwise. */

/* Optional weight tying (-DTIE_WEIGHTS): reuse the input embedding as the
 * output projection (a classic LM trick — the embedding gets gradient signal
 * from both the input and the output, usually improving it). It requires the
 * embedding to be hidden-width so the shapes match; no new arrays, no change
 * to the checkpoint format (the now-unused Why is just ignored). */
#ifdef TIE_WEIGHTS
# if EMBED != HIDDEN
#  error "TIE_WEIGHTS requires EMBED == HIDDEN (e.g. -DHIDDEN=256 -DEMBED=256)"
# endif
#endif

#ifdef USE_LN
#define CKPT_MAGIC  0x534C4D34u   /* "SLM4" — GRU + Adam + LayerNorm gains */
#else
#define CKPT_MAGIC  0x534C4D33u   /* "SLM3" — GRU + Adam checkpoint format */
#endif
#ifndef CKPT_PATH
#define CKPT_PATH   "sentinel.bin"
#endif
#ifndef CKPT_EVERY
#define CKPT_EVERY  2000    /* auto-save every N iters of a long training run, */
#endif                      /* so an interrupted multi-hour train isn't lost   */
#define VERSION     "0.6.0"
/* Optional build flags (all default OFF; the shipped model is unchanged):
 *   -DUSE_LN        LayerNorm on the GRU gate pre-activations (faster/steadier
 *                   convergence; adds a learnable gain gln, reuses bg as beta).
 *   -DTIE_WEIGHTS   tie the output projection to the input embedding (needs
 *                   EMBED==HIDDEN). Both are gradient-checked (--gradcheck). */

/* ------------------------------------------------------------------ */
/*  Parameters — a stacked GRU (gated recurrent unit) network.         */
/*                                                                     */
/*  Each layer has three gates: z (update), r (reset), n (candidate).  */
/*  Per gate there is an input weight Wg, a recurrent weight Ug and a   */
/*  bias bg.  Gates let the layer keep or forget hidden state, which    */
/*  fixes the vanishing-gradient problem of a plain tanh RNN and lets   */
/*  depth + long sequences actually help.                              */
/*  EMBED <= HIDDEN, so layer-0 input is zero-padded up to HIDDEN and   */
/*  every weight matrix shares a uniform HIDDEN-wide stride.            */
/* ------------------------------------------------------------------ */
enum { GZ = 0, GR = 1, GN = 2, NGATE = 3 };       /* update / reset / candidate */

static double Wemb[VOCAB][EMBED];                          /* embedding table */
static double Wg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];     /* input  -> gate  */
static double Ug  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];     /* hidden -> gate  */
static double bg  [NUM_LAYERS][NGATE][HIDDEN];             /* gate bias       */
static double Why [VOCAB][HIDDEN];                          /* hidden -> output*/
static double by  [VOCAB];                                  /* output bias     */

static double dWemb[VOCAB][EMBED];
static double dWg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double dUg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double dbg  [NUM_LAYERS][NGATE][HIDDEN];
static double dWhy [VOCAB][HIDDEN];
static double dby  [VOCAB];

/* Adam first-moment (m) and second-moment (v) estimates per weight. */
static double mWemb[VOCAB][EMBED];
static double mWg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double mUg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double mbg  [NUM_LAYERS][NGATE][HIDDEN];
static double mWhy [VOCAB][HIDDEN];
static double mby  [VOCAB];

static double vWemb[VOCAB][EMBED];
static double vWg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double vUg  [NUM_LAYERS][NGATE][HIDDEN][HIDDEN];
static double vbg  [NUM_LAYERS][NGATE][HIDDEN];
static double vWhy [VOCAB][HIDDEN];
static double vby  [VOCAB];

#ifdef USE_LN
/* LayerNorm gain (gamma) per gate channel; the existing bg acts as the beta
 * (post-norm shift). Added when built with -DUSE_LN. */
static double gln [NUM_LAYERS][NGATE][HIDDEN];
static double dgln[NUM_LAYERS][NGATE][HIDDEN];
static double mgln[NUM_LAYERS][NGATE][HIDDEN];
static double vgln[NUM_LAYERS][NGATE][HIDDEN];
#endif

static long g_adam_t = 0;   /* Adam timestep, for bias correction */
static double g_lr = LR;    /* effective learning rate (override via SLM_LR) */
static int  g_defer_adam = 0;  /* when set, train_step computes grads but skips Adam
                                * (used by quantization-aware training) */

/* Per-timestep activation cache used by backprop-through-time. */
static int    g_in [SEQ_LEN];
static double g_emb[SEQ_LEN][EMBED];
static double g_h  [NUM_LAYERS][SEQ_LEN + 1][HIDDEN];  /* [l][0] = carried state */
static double g_z  [NUM_LAYERS][SEQ_LEN][HIDDEN];      /* update gate           */
static double g_r  [NUM_LAYERS][SEQ_LEN][HIDDEN];      /* reset  gate           */
static double g_n  [NUM_LAYERS][SEQ_LEN][HIDDEN];      /* candidate state       */
static double g_hr [NUM_LAYERS][SEQ_LEN][HIDDEN];      /* r (.) h_prev          */
static double g_y  [SEQ_LEN][VOCAB];
static double g_p  [SEQ_LEN][VOCAB];
#ifdef USE_LN
static double g_ln_nhat[NUM_LAYERS][NGATE][SEQ_LEN][HIDDEN];  /* normalized pre-act */
static double g_ln_rstd[NUM_LAYERS][NGATE][SEQ_LEN];          /* 1/std per gate/step */
#endif

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

/* Logistic sigmoid, used by the GRU update/reset gates.  Branch on sign so a
 * large-magnitude input never overflows exp() (avoids a spurious FP flag). */
static double sigmoid(double x) {
    if (x >= 0.0) return 1.0 / (1.0 + exp(-x));
    double e = exp(x);
    return e / (1.0 + e);
}

/* ------------------------------------------------------------------ */
/*  Weight initialisation (fixed seed -> reproducible static weights). */
/* ------------------------------------------------------------------ */
static void init_weights(void) {
    rng_state = 0x9E3779B97F4A7C15ULL;
    /* Xavier-ish scale keeps gate pre-activations sane at this width. */
    double s = 1.0 / sqrt((double)HIDDEN);
    for (int t = 0; t < VOCAB; t++)
        for (int j = 0; j < EMBED; j++)
            Wemb[t][j] = rnd_uniform(-0.1, 0.1);
    for (int l = 0; l < NUM_LAYERS; l++) {
        for (int gt = 0; gt < NGATE; gt++) {
            for (int i = 0; i < HIDDEN; i++) {
                for (int j = 0; j < HIDDEN; j++) {
                    Wg[l][gt][i][j] = rnd_uniform(-s, s);
                    Ug[l][gt][i][j] = rnd_uniform(-s, s);
                }
                bg[l][gt][i] = 0.0;
            }
        }
    }
    for (int i = 0; i < VOCAB; i++) {
        for (int j = 0; j < HIDDEN; j++) Why[i][j] = rnd_uniform(-0.1, 0.1);
        by[i] = 0.0;
    }
    memset(mWemb, 0, sizeof mWemb); memset(mWg, 0, sizeof mWg);
    memset(mUg, 0, sizeof mUg);     memset(mbg, 0, sizeof mbg);
    memset(mWhy, 0, sizeof mWhy);   memset(mby, 0, sizeof mby);
    memset(vWemb, 0, sizeof vWemb); memset(vWg, 0, sizeof vWg);
    memset(vUg, 0, sizeof vUg);     memset(vbg, 0, sizeof vbg);
    memset(vWhy, 0, sizeof vWhy);   memset(vby, 0, sizeof vby);
#ifdef USE_LN
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int gt = 0; gt < NGATE; gt++)
            for (int i = 0; i < HIDDEN; i++) gln[l][gt][i] = 1.0;   /* identity scale */
    memset(mgln, 0, sizeof mgln); memset(vgln, 0, sizeof vgln);
#endif
    g_adam_t = 0;
}

/* ------------------------------------------------------------------ */
/*  One BPTT step over a SEQ_LEN window.  Returns the cross-entropy    */
/*  loss; updates weights in place (Adam) and carries hidden state.    */
/* ------------------------------------------------------------------ */
/* Adam update with bias correction.  bc1 = 1-B1^t, bc2 = 1-B2^t are passed in
 * so they're computed once per step, not once per weight. */
static void adam(double *p, double *g, double *m, double *v, int n,
                 double bc1, double bc2, double lr) {
    for (int i = 0; i < n; i++) {
        double grad = g[i];
        if (grad >  CLIP) grad =  CLIP;
        if (grad < -CLIP) grad = -CLIP;
        m[i] = ADAM_B1 * m[i] + (1.0 - ADAM_B1) * grad;
        v[i] = ADAM_B2 * v[i] + (1.0 - ADAM_B2) * grad * grad;
        double mhat = m[i] / bc1;
        double vhat = v[i] / bc2;
        p[i] -= lr * mhat / (sqrt(vhat) + ADAM_EPS);
    }
}

/* Backprop one GRU gate: accumulate dWg/dUg/dbg, add input grad into dxin,
 * and return the recurrent grad drec = Ug^T * d (gradient w.r.t. the gate's
 * recurrent input vector). */
static void gate_backward(int l, int gt, const double *d, const double *xin,
                          const double *rec, double *dxin, double *drec) {
    for (int i = 0; i < HIDDEN; i++) {
        double di = d[i];
#ifndef USE_LN
        dbg[l][gt][i] += di;        /* with LN the bias grad is handled post-norm */
#endif
        double *wr = &dWg[l][gt][i][0];
        double *ur = &dUg[l][gt][i][0];
        for (int j = 0; j < HIDDEN; j++) { wr[j] += di * xin[j]; ur[j] += di * rec[j]; }
    }
    for (int j = 0; j < HIDDEN; j++) {
        double sx = 0.0, sr = 0.0;
        for (int i = 0; i < HIDDEN; i++) {
            sx += Wg[l][gt][i][j] * d[i];
            sr += Ug[l][gt][i][j] * d[i];
        }
        dxin[j] += sx;
        drec[j]  = sr;
    }
}

/* Apply one Adam step to every parameter from the accumulated gradients.
 * Split out of train_step so quantization-aware training can run the
 * forward/backward on quantized weights but apply Adam to the latent ones. */
static void apply_adam(void) {
    g_adam_t++;
    double bc1 = 1.0 - pow(ADAM_B1, (double)g_adam_t);
    double bc2 = 1.0 - pow(ADAM_B2, (double)g_adam_t);
    /* linear warmup for a fresh model (g_adam_t small); a resumed checkpoint
     * has a large g_adam_t, so it trains at full LR immediately. */
    double lr = (g_adam_t < WARMUP) ? g_lr * (double)g_adam_t / (double)WARMUP : g_lr;
    adam(&Wemb[0][0], &dWemb[0][0], &mWemb[0][0], &vWemb[0][0], VOCAB * EMBED, bc1, bc2, lr);
    adam(&Why[0][0],  &dWhy[0][0],  &mWhy[0][0],  &vWhy[0][0],  VOCAB * HIDDEN, bc1, bc2, lr);
    adam(by, dby, mby, vby, VOCAB, bc1, bc2, lr);
    for (int l = 0; l < NUM_LAYERS; l++) {
        adam(&Wg[l][0][0][0], &dWg[l][0][0][0], &mWg[l][0][0][0], &vWg[l][0][0][0],
             NGATE * HIDDEN * HIDDEN, bc1, bc2, lr);
        adam(&Ug[l][0][0][0], &dUg[l][0][0][0], &mUg[l][0][0][0], &vUg[l][0][0][0],
             NGATE * HIDDEN * HIDDEN, bc1, bc2, lr);
        adam(&bg[l][0][0], &dbg[l][0][0], &mbg[l][0][0], &vbg[l][0][0], NGATE * HIDDEN, bc1, bc2, lr);
    }
#ifdef USE_LN
    adam(&gln[0][0][0], &dgln[0][0][0], &mgln[0][0][0], &vgln[0][0][0],
         NUM_LAYERS * NGATE * HIDDEN, bc1, bc2, lr);
#endif
}

#ifdef USE_LN
/* LayerNorm over a HIDDEN vector. Fills nhat=(v-μ)/σ, returns rstd=1/σ. */
static double ln_forward(const double *v, double *nhat) {
    double mu = 0.0;
    for (int i = 0; i < HIDDEN; i++) mu += v[i];
    mu /= HIDDEN;
    double var = 0.0;
    for (int i = 0; i < HIDDEN; i++) { double d = v[i] - mu; var += d * d; }
    var /= HIDDEN;
    double rstd = 1.0 / sqrt(var + 1e-5);
    for (int i = 0; i < HIDDEN; i++) nhat[i] = (v[i] - mu) * rstd;
    return rstd;
}
/* LayerNorm backward: given dnhat and cached nhat/rstd, fill dv (grad wrt v). */
static void ln_backward(const double *dnhat, const double *nhat, double rstd, double *dv) {
    double m1 = 0.0, m2 = 0.0;
    for (int i = 0; i < HIDDEN; i++) { m1 += dnhat[i]; m2 += dnhat[i] * nhat[i]; }
    m1 /= HIDDEN; m2 /= HIDDEN;
    for (int i = 0; i < HIDDEN; i++) dv[i] = rstd * (dnhat[i] - m1 - nhat[i] * m2);
}
/* Turn d_a (grad wrt the post-norm activation a=γ⊙nhat+β) into dpre (grad wrt
 * the pre-norm vector), accumulating the β (bg) and γ (gln) gradients. */
static void ln_gate_back(int l, int gt, int t, const double *da, double *dpre) {
    double dnhat[HIDDEN];
    const double *nhat = g_ln_nhat[l][gt][t];
    for (int i = 0; i < HIDDEN; i++) {
        dbg[l][gt][i]  += da[i];
        dgln[l][gt][i] += da[i] * nhat[i];
        dnhat[i] = da[i] * gln[l][gt][i];
    }
    ln_backward(dnhat, nhat, g_ln_rstd[l][gt][t], dpre);
}
#endif

static double train_step(const int *inputs, const int *targets,
                         double hprev[NUM_LAYERS][HIDDEN]) {
    /* ---- forward ---- */
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int i = 0; i < HIDDEN; i++) g_h[l][0][i] = hprev[l][i];

    double loss = 0.0;
    double xin[HIDDEN], wx[HIDDEN], uh[HIDDEN];
#ifdef USE_LN
    double pre[HIDDEN];
#endif
    for (int t = 0; t < SEQ_LEN; t++) {
        int tok = inputs[t];
        g_in[t] = tok;
        for (int j = 0; j < EMBED; j++) g_emb[t][j] = Wemb[tok][j];

        for (int l = 0; l < NUM_LAYERS; l++) {
            const double *hp = g_h[l][t];
            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = g_emb[t][j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = g_h[l - 1][t + 1][j];
            }
            /* update gate z */
            matvec(wx, &Wg[l][GZ][0][0], xin, HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GZ][0][0], hp,  HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            g_ln_rstd[l][GZ][t] = ln_forward(pre, g_ln_nhat[l][GZ][t]);
            for (int i = 0; i < HIDDEN; i++)
                g_z[l][t][i] = sigmoid(gln[l][GZ][i] * g_ln_nhat[l][GZ][t][i] + bg[l][GZ][i]);
#else
            for (int i = 0; i < HIDDEN; i++)
                g_z[l][t][i] = sigmoid(wx[i] + uh[i] + bg[l][GZ][i]);
#endif
            /* reset gate r */
            matvec(wx, &Wg[l][GR][0][0], xin, HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GR][0][0], hp,  HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            g_ln_rstd[l][GR][t] = ln_forward(pre, g_ln_nhat[l][GR][t]);
            for (int i = 0; i < HIDDEN; i++)
                g_r[l][t][i] = sigmoid(gln[l][GR][i] * g_ln_nhat[l][GR][t][i] + bg[l][GR][i]);
#else
            for (int i = 0; i < HIDDEN; i++)
                g_r[l][t][i] = sigmoid(wx[i] + uh[i] + bg[l][GR][i]);
#endif
            /* candidate n with reset-gated hidden: hr = r (.) h_prev */
            for (int i = 0; i < HIDDEN; i++) g_hr[l][t][i] = g_r[l][t][i] * hp[i];
            matvec(wx, &Wg[l][GN][0][0], xin,        HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GN][0][0], g_hr[l][t], HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            g_ln_rstd[l][GN][t] = ln_forward(pre, g_ln_nhat[l][GN][t]);
            for (int i = 0; i < HIDDEN; i++)
                g_n[l][t][i] = tanh(gln[l][GN][i] * g_ln_nhat[l][GN][t][i] + bg[l][GN][i]);
#else
            for (int i = 0; i < HIDDEN; i++)
                g_n[l][t][i] = tanh(wx[i] + uh[i] + bg[l][GN][i]);
#endif
            /* new hidden: h = (1 - z) (.) n + z (.) h_prev */
            for (int i = 0; i < HIDDEN; i++)
                g_h[l][t + 1][i] = (1.0 - g_z[l][t][i]) * g_n[l][t][i]
                                   + g_z[l][t][i] * hp[i];
        }
#ifdef TIE_WEIGHTS
        matvec(g_y[t], &Wemb[0][0], g_h[NUM_LAYERS - 1][t + 1], VOCAB, HIDDEN);
#else
        matvec(g_y[t], &Why[0][0], g_h[NUM_LAYERS - 1][t + 1], VOCAB, HIDDEN);
#endif
        for (int i = 0; i < VOCAB; i++) g_y[t][i] += by[i];
        softmax(g_p[t], g_y[t], VOCAB);
        loss += -log(g_p[t][targets[t]] + 1e-12);
    }

    /* ---- backward (through depth and time) ---- */
    memset(dWemb, 0, sizeof dWemb); memset(dWg, 0, sizeof dWg);
    memset(dUg, 0, sizeof dUg);     memset(dbg, 0, sizeof dbg);
    memset(dWhy, 0, sizeof dWhy);   memset(dby, 0, sizeof dby);
#ifdef USE_LN
    memset(dgln, 0, sizeof dgln);
#endif

    double dh_next[NUM_LAYERS][HIDDEN];
    memset(dh_next, 0, sizeof dh_next);

    for (int t = SEQ_LEN - 1; t >= 0; t--) {
        double dy[VOCAB];
        for (int i = 0; i < VOCAB; i++) dy[i] = g_p[t][i];
        dy[targets[t]] -= 1.0;

        const double *top = g_h[NUM_LAYERS - 1][t + 1];
        for (int i = 0; i < VOCAB; i++) {
            double dyi = dy[i];
            dby[i] += dyi;
#ifdef TIE_WEIGHTS
            double *wr = &dWemb[i][0];   /* output grad flows into the embedding */
#else
            double *wr = &dWhy[i][0];
#endif
            for (int j = 0; j < HIDDEN; j++) wr[j] += dyi * top[j];
        }

        double dhtot[NUM_LAYERS][HIDDEN];
        for (int l = 0; l < NUM_LAYERS; l++)
            for (int i = 0; i < HIDDEN; i++) dhtot[l][i] = dh_next[l][i];
        for (int j = 0; j < HIDDEN; j++) {
            double s = 0.0;
#ifdef TIE_WEIGHTS
            for (int i = 0; i < VOCAB; i++) s += Wemb[i][j] * dy[i];
#else
            for (int i = 0; i < VOCAB; i++) s += Why[i][j] * dy[i];
#endif
            dhtot[NUM_LAYERS - 1][j] += s;
        }

        for (int l = NUM_LAYERS - 1; l >= 0; l--) {
            const double *hp = g_h[l][t];
            const double *z  = g_z[l][t];
            const double *r  = g_r[l][t];
            const double *nn = g_n[l][t];
            const double *hr = g_hr[l][t];
            double *dh = dhtot[l];

            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = g_emb[t][j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = g_h[l - 1][t + 1][j];
            }

            double dn[HIDDEN], dz[HIDDEN], dhp[HIDDEN];
            double dan[HIDDEN], daz[HIDDEN], dar[HIDDEN], dr[HIDDEN];
            double dxin[HIDDEN], drec[HIDDEN];
#ifdef USE_LN
            double dpre[HIDDEN];
#endif
            for (int i = 0; i < HIDDEN; i++) {
                dn[i]   = dh[i] * (1.0 - z[i]);
                dz[i]   = dh[i] * (hp[i] - nn[i]);
                dhp[i]  = dh[i] * z[i];          /* direct h_prev path */
                dxin[i] = 0.0;
            }
            /* candidate n = tanh(Wn x + Un hr + bn) */
            for (int i = 0; i < HIDDEN; i++) dan[i] = dn[i] * (1.0 - nn[i] * nn[i]);
#ifdef USE_LN
            ln_gate_back(l, GN, t, dan, dpre);
            gate_backward(l, GN, dpre, xin, hr, dxin, drec);  /* drec = grad wrt hr */
#else
            gate_backward(l, GN, dan, xin, hr, dxin, drec);   /* drec = grad wrt hr */
#endif
            for (int i = 0; i < HIDDEN; i++) {
                dr[i]   = drec[i] * hp[i];       /* hr = r (.) h_prev */
                dhp[i] += drec[i] * r[i];
            }
            /* update gate z = sigmoid(...) */
            for (int i = 0; i < HIDDEN; i++) daz[i] = dz[i] * z[i] * (1.0 - z[i]);
#ifdef USE_LN
            ln_gate_back(l, GZ, t, daz, dpre);
            gate_backward(l, GZ, dpre, xin, hp, dxin, drec);
#else
            gate_backward(l, GZ, daz, xin, hp, dxin, drec);
#endif
            for (int i = 0; i < HIDDEN; i++) dhp[i] += drec[i];
            /* reset gate r = sigmoid(...) */
            for (int i = 0; i < HIDDEN; i++) dar[i] = dr[i] * r[i] * (1.0 - r[i]);
#ifdef USE_LN
            ln_gate_back(l, GR, t, dar, dpre);
            gate_backward(l, GR, dpre, xin, hp, dxin, drec);
#else
            gate_backward(l, GR, dar, xin, hp, dxin, drec);
#endif
            for (int i = 0; i < HIDDEN; i++) dhp[i] += drec[i];

            /* temporal gradient for t-1, and route the input gradient down */
            for (int i = 0; i < HIDDEN; i++) dh_next[l][i] = dhp[i];
            if (l == 0) {
                int tok = g_in[t];
                for (int j = 0; j < EMBED; j++) dWemb[tok][j] += dxin[j];
            } else {
                for (int j = 0; j < HIDDEN; j++) dhtot[l - 1][j] += dxin[j];
            }
        }
    }

    /* ---- Adam update (deferred during quantization-aware training) ---- */
    if (!g_defer_adam) apply_adam();

    /* carry the last hidden state forward (continuous, no context limit) */
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int i = 0; i < HIDDEN; i++) hprev[l][i] = g_h[l][SEQ_LEN][i];
    return loss;
}

/* ------------------------------------------------------------------ */
/*  Train on a text buffer for a number of iterations.                 */
/* ------------------------------------------------------------------ */
static void save_model(const char *path);   /* defined below; used for auto-save */

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
        if (verbose && it > 0 && it % CKPT_EVERY == 0) {
            save_model(CKPT_PATH);          /* crash-safety for long runs */
            printf("  [auto-saved checkpoint at iter %d]\n", it);
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Numerical gradient check — verify backprop against finite diffs.    */
/*  Proves the hand-written gradients (and any new ones: tying, LN) are */
/*  correct. Build tiny (e.g. HIDDEN=16 EMBED=8) and run --gradcheck.   */
/* ------------------------------------------------------------------ */
static double forward_loss(const int *inputs, const int *targets,
                           double hprev0[NUM_LAYERS][HIDDEN]) {
    double hp[NUM_LAYERS][HIDDEN];
    memcpy(hp, hprev0, sizeof hp);
    g_defer_adam = 1;
    double L = train_step(inputs, targets, hp);   /* no weight update */
    g_defer_adam = 0;
    return L;
}

static void gradcheck(void) {
    const char seed[] =
        "the local slm studies cybersecurity and coding; a secure system validates input.";
    if ((int)(sizeof seed - 1) < SEQ_LEN + 1) { printf("gradcheck: seed too short\n"); return; }
    int inputs[SEQ_LEN], targets[SEQ_LEN];
    for (int k = 0; k < SEQ_LEN; k++) {
        inputs[k]  = (unsigned char)seed[k]     & (VOCAB - 1);
        targets[k] = (unsigned char)seed[k + 1] & (VOCAB - 1);
    }
    double hprev0[NUM_LAYERS][HIDDEN];
    memset(hprev0, 0, sizeof hprev0);

    /* analytic gradients from one backward pass */
    double hp[NUM_LAYERS][HIDDEN];
    memcpy(hp, hprev0, sizeof hp);
    g_defer_adam = 1;
    train_step(inputs, targets, hp);
    g_defer_adam = 0;

    struct { double *w; double g; const char *name; } probes[] = {
        { &Wg[0][GZ][0][0], dWg[0][GZ][0][0], "Wg[0][GZ][0][0]" },
        { &Wg[0][GN][1][2], dWg[0][GN][1][2], "Wg[0][GN][1][2]" },
        { &Ug[0][GR][2][1], dUg[0][GR][2][1], "Ug[0][GR][2][1]" },
        { &Wemb[inputs[0]][0], dWemb[inputs[0]][0], "Wemb[tok][0]" },
        { &Why[1][3], dWhy[1][3], "Why[1][3]" },
        { &by[2], dby[2], "by[2]" },
        { &bg[0][GZ][4], dbg[0][GZ][4], "bg[0][GZ][4]" },
#ifdef USE_LN
        { &gln[0][GN][3], dgln[0][GN][3], "gln[0][GN][3]" },
        { &gln[1][GZ][2], dgln[1][GZ][2], "gln[1][GZ][2]" },
#endif
    };
    int np = (int)(sizeof probes / sizeof probes[0]);
    double eps = 1e-5;
    int fail = 0;
    printf("gradcheck (eps=%.0e):\n", eps);
    for (int i = 0; i < np; i++) {
        double *w = probes[i].w, orig = *w, a = probes[i].g;
        *w = orig + eps; double Lp = forward_loss(inputs, targets, hprev0);
        *w = orig - eps; double Lm = forward_loss(inputs, targets, hprev0);
        *w = orig;
        double num = (Lp - Lm) / (2.0 * eps);
        double adiff = fabs(a - num);
        double rel = adiff / (fabs(a) + fabs(num) + 1e-12);
        /* a gradient is wrong only if BOTH the relative and absolute error are
         * large; near-zero gradients fail relative error on FP noise alone. */
        int bad = (rel > 1e-4 && adiff > 1e-7);
        if (bad) fail = 1;
        printf("  %-18s analytic=% .6e  numeric=% .6e  relerr=%.2e%s\n",
               probes[i].name, a, num, rel, bad ? "  <-- BAD" : "");
    }
    printf("gradcheck  ->  %s\n", fail ? "FAIL" : "PASS");
}

/* ------------------------------------------------------------------ */
/*  Generate text by sampling the model character by character.        */
/* ------------------------------------------------------------------ */
static void sample(int seed, int n) {
    double h[NUM_LAYERS][HIDDEN], hnew[NUM_LAYERS][HIDDEN];
    memset(h, 0, sizeof h);
    double xin[HIDDEN], wx[HIDDEN], uh[HIDDEN], hr[HIDDEN];
    double z[HIDDEN], rr[HIDDEN], nn[HIDDEN], emb[EMBED], y[VOCAB], p[VOCAB];
#ifdef USE_LN
    double pre[HIDDEN], nh[HIDDEN];
#endif
    int tok = seed & (VOCAB - 1);

    for (int step = 0; step < n; step++) {
        for (int j = 0; j < EMBED; j++) emb[j] = Wemb[tok][j];
        for (int l = 0; l < NUM_LAYERS; l++) {
            const double *hp = h[l];
            if (l == 0) {
                for (int j = 0; j < EMBED;  j++) xin[j] = emb[j];
                for (int j = EMBED; j < HIDDEN; j++) xin[j] = 0.0;
            } else {
                for (int j = 0; j < HIDDEN; j++) xin[j] = hnew[l - 1][j];
            }
            matvec(wx, &Wg[l][GZ][0][0], xin, HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GZ][0][0], hp,  HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            ln_forward(pre, nh);
            for (int i = 0; i < HIDDEN; i++) z[i] = sigmoid(gln[l][GZ][i] * nh[i] + bg[l][GZ][i]);
#else
            for (int i = 0; i < HIDDEN; i++) z[i]  = sigmoid(wx[i] + uh[i] + bg[l][GZ][i]);
#endif
            matvec(wx, &Wg[l][GR][0][0], xin, HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GR][0][0], hp,  HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            ln_forward(pre, nh);
            for (int i = 0; i < HIDDEN; i++) rr[i] = sigmoid(gln[l][GR][i] * nh[i] + bg[l][GR][i]);
#else
            for (int i = 0; i < HIDDEN; i++) rr[i] = sigmoid(wx[i] + uh[i] + bg[l][GR][i]);
#endif
            for (int i = 0; i < HIDDEN; i++) hr[i] = rr[i] * hp[i];
            matvec(wx, &Wg[l][GN][0][0], xin, HIDDEN, HIDDEN);
            matvec(uh, &Ug[l][GN][0][0], hr,  HIDDEN, HIDDEN);
#ifdef USE_LN
            for (int i = 0; i < HIDDEN; i++) pre[i] = wx[i] + uh[i];
            ln_forward(pre, nh);
            for (int i = 0; i < HIDDEN; i++) nn[i] = tanh(gln[l][GN][i] * nh[i] + bg[l][GN][i]);
#else
            for (int i = 0; i < HIDDEN; i++) nn[i] = tanh(wx[i] + uh[i] + bg[l][GN][i]);
#endif
            for (int i = 0; i < HIDDEN; i++)
                hnew[l][i] = (1.0 - z[i]) * nn[i] + z[i] * hp[i];
        }
        for (int l = 0; l < NUM_LAYERS; l++)
            for (int i = 0; i < HIDDEN; i++) h[l][i] = hnew[l][i];
#ifdef TIE_WEIGHTS
        matvec(y, &Wemb[0][0], h[NUM_LAYERS - 1], VOCAB, HIDDEN);
#else
        matvec(y, &Why[0][0], h[NUM_LAYERS - 1], VOCAB, HIDDEN);
#endif
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
    fwrite(Wg,   sizeof Wg,   1, f);
    fwrite(Ug,   sizeof Ug,   1, f);
    fwrite(bg,   sizeof bg,   1, f);
    fwrite(Why,  sizeof Why,  1, f);
    fwrite(by,   sizeof by,   1, f);
    fwrite(mWemb, sizeof mWemb, 1, f);
    fwrite(mWg,   sizeof mWg,   1, f);
    fwrite(mUg,   sizeof mUg,   1, f);
    fwrite(mbg,   sizeof mbg,   1, f);
    fwrite(mWhy,  sizeof mWhy,  1, f);
    fwrite(mby,   sizeof mby,   1, f);
    fwrite(vWemb, sizeof vWemb, 1, f);
    fwrite(vWg,   sizeof vWg,   1, f);
    fwrite(vUg,   sizeof vUg,   1, f);
    fwrite(vbg,   sizeof vbg,   1, f);
    fwrite(vWhy,  sizeof vWhy,  1, f);
    fwrite(vby,   sizeof vby,   1, f);
#ifdef USE_LN
    fwrite(gln,  sizeof gln,  1, f);
    fwrite(mgln, sizeof mgln, 1, f);
    fwrite(vgln, sizeof vgln, 1, f);
#endif
    fwrite(&g_adam_t, sizeof g_adam_t, 1, f);
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
    ok &= fread(Wg,   sizeof Wg,   1, f) == 1;
    ok &= fread(Ug,   sizeof Ug,   1, f) == 1;
    ok &= fread(bg,   sizeof bg,   1, f) == 1;
    ok &= fread(Why,  sizeof Why,  1, f) == 1;
    ok &= fread(by,   sizeof by,   1, f) == 1;
    ok &= fread(mWemb, sizeof mWemb, 1, f) == 1;
    ok &= fread(mWg,   sizeof mWg,   1, f) == 1;
    ok &= fread(mUg,   sizeof mUg,   1, f) == 1;
    ok &= fread(mbg,   sizeof mbg,   1, f) == 1;
    ok &= fread(mWhy,  sizeof mWhy,  1, f) == 1;
    ok &= fread(mby,   sizeof mby,   1, f) == 1;
    ok &= fread(vWemb, sizeof vWemb, 1, f) == 1;
    ok &= fread(vWg,   sizeof vWg,   1, f) == 1;
    ok &= fread(vUg,   sizeof vUg,   1, f) == 1;
    ok &= fread(vbg,   sizeof vbg,   1, f) == 1;
    ok &= fread(vWhy,  sizeof vWhy,  1, f) == 1;
    ok &= fread(vby,   sizeof vby,   1, f) == 1;
#ifdef USE_LN
    ok &= fread(gln,  sizeof gln,  1, f) == 1;
    ok &= fread(mgln, sizeof mgln, 1, f) == 1;
    ok &= fread(vgln, sizeof vgln, 1, f) == 1;
#endif
    ok &= fread(&g_adam_t, sizeof g_adam_t, 1, f) == 1;
    fclose(f);
    return ok;
}

/* ================================================================== */
/*  Weight quantization — ship a SMALL model, keep the float one safe.  */
/*                                                                     */
/*  The trained weights live as 8-byte doubles (~1.85 GB at the 58M    */
/*  default).  That is huge to ship to a Pi / USB stick / ESP32.  These */
/*  routines export a compact quantized copy of the big weight matrices */
/*  WITHOUT touching the float checkpoint:                              */
/*                                                                     */
/*    int8  — 1 byte/param, per-row symmetric scale.  ~8x smaller than  */
/*            the double checkpoint, keeps ~all the quality.            */
/*    1-bit — 1 bit/param (the sign) + a per-row magnitude alpha        */
/*            (BinaryConnect / XNOR-Net style).  ~64x smaller; lossy,   */
/*            but small enough to fit a bigger model on tiny hardware.  */
/*                                                                     */
/*  Biases and the embedding stay full/float precision — they are tiny  */
/*  and quality-critical.  Quantization is done ROW BY ROW into a small */
/*  heap buffer, so no large extra array is ever allocated (the default */
/*  build must stay under the 2 GB static-data limit).                  */
/* ================================================================== */
#define QUANT_MAGIC 0x534C4D51u   /* "SLMQ" — quantized export format */

/* Write one row-major tensor (rows x cols) to f, quantized per row. */
static int quant_write_tensor(FILE *f, const double *M, int rows, int cols, int qmode) {
    signed char   *q8 = (qmode == 8) ? malloc((size_t)cols)         : NULL;
    unsigned char *q1 = (qmode == 1) ? malloc(((size_t)cols + 7) / 8) : NULL;
    if ((qmode == 8 && !q8) || (qmode == 1 && !q1)) { free(q8); free(q1); return 0; }
    for (int i = 0; i < rows; i++) {
        const double *row = M + (size_t)i * cols;
        if (qmode == 8) {
            double mx = 0.0;
            for (int j = 0; j < cols; j++) { double a = fabs(row[j]); if (a > mx) mx = a; }
            float scale = (mx > 0.0) ? (float)(mx / 127.0) : 1.0f;
            fwrite(&scale, sizeof scale, 1, f);
            for (int j = 0; j < cols; j++) {
                long v = lround(row[j] / scale);
                if (v >  127) v =  127;
                if (v < -127) v = -127;
                q8[j] = (signed char)v;
            }
            fwrite(q8, 1, (size_t)cols, f);
        } else {
            double s = 0.0;
            for (int j = 0; j < cols; j++) s += fabs(row[j]);
            float alpha = (cols > 0 && s > 0.0) ? (float)(s / cols) : 1.0f;
            fwrite(&alpha, sizeof alpha, 1, f);
            size_t nb = ((size_t)cols + 7) / 8;
            memset(q1, 0, nb);
            for (int j = 0; j < cols; j++)
                if (row[j] >= 0.0) q1[j >> 3] |= (unsigned char)(1u << (j & 7));
            fwrite(q1, 1, nb, f);
        }
    }
    free(q8); free(q1);
    return 1;
}

/* Read one quantized tensor back, dequantizing into the double array M. */
static int quant_read_tensor(FILE *f, double *M, int rows, int cols, int qmode) {
    signed char   *q8 = (qmode == 8) ? malloc((size_t)cols)         : NULL;
    unsigned char *q1 = (qmode == 1) ? malloc(((size_t)cols + 7) / 8) : NULL;
    if ((qmode == 8 && !q8) || (qmode == 1 && !q1)) { free(q8); free(q1); return 0; }
    int ok = 1;
    for (int i = 0; i < rows && ok; i++) {
        double *row = M + (size_t)i * cols;
        if (qmode == 8) {
            float scale;
            ok &= fread(&scale, sizeof scale, 1, f) == 1;
            ok &= fread(q8, 1, (size_t)cols, f) == (size_t)cols;
            for (int j = 0; j < cols && ok; j++) row[j] = (double)q8[j] * scale;
        } else {
            float alpha;
            size_t nb = ((size_t)cols + 7) / 8;
            ok &= fread(&alpha, sizeof alpha, 1, f) == 1;
            ok &= fread(q1, 1, nb, f) == nb;
            for (int j = 0; j < cols && ok; j++)
                row[j] = (q1[j >> 3] & (1u << (j & 7))) ? (double)alpha : -(double)alpha;
        }
    }
    free(q8); free(q1);
    return ok;
}

/* Small full-precision tensors (biases) are stored as float32. */
static void f32_write(FILE *f, const double *a, int n) {
    for (int i = 0; i < n; i++) { float v = (float)a[i]; fwrite(&v, sizeof v, 1, f); }
}
static int f32_read(FILE *f, double *a, int n) {
    int ok = 1;
    for (int i = 0; i < n; i++) { float v; ok &= fread(&v, sizeof v, 1, f) == 1; a[i] = v; }
    return ok;
}

/* Export a quantized copy of the current weights.  Refuses to overwrite the
 * float checkpoint, so a trained model can never be clobbered by accident. */
static int save_quant(const char *path, int qmode) {
    if (strcmp(path, CKPT_PATH) == 0) {
        fprintf(stderr, "quantize: refusing to overwrite the float checkpoint '%s'\n", path);
        return 0;
    }
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    unsigned hdr[7] = { QUANT_MAGIC, (unsigned)qmode, VOCAB, EMBED, HIDDEN, NUM_LAYERS, 1u };
    fwrite(hdr, sizeof(unsigned), 7, f);
    int ok = 1;
    ok &= quant_write_tensor(f, &Wemb[0][0], VOCAB, EMBED, qmode);
    for (int l = 0; l < NUM_LAYERS && ok; l++)
        for (int g = 0; g < NGATE && ok; g++) {
            ok &= quant_write_tensor(f, &Wg[l][g][0][0], HIDDEN, HIDDEN, qmode);
            ok &= quant_write_tensor(f, &Ug[l][g][0][0], HIDDEN, HIDDEN, qmode);
        }
    ok &= quant_write_tensor(f, &Why[0][0], VOCAB, HIDDEN, qmode);
    f32_write(f, &bg[0][0][0], NUM_LAYERS * NGATE * HIDDEN);   /* biases: full precision */
    f32_write(f, by, VOCAB);
    fclose(f);
    return ok;
}

/* Load a quantized export, dequantizing into the live weight arrays so the
 * existing sample()/forward path can run it unchanged. */
static int load_quant(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    unsigned hdr[7];
    if (fread(hdr, sizeof(unsigned), 7, f) != 7 || hdr[0] != QUANT_MAGIC ||
        hdr[2] != VOCAB || hdr[3] != EMBED || hdr[4] != HIDDEN || hdr[5] != NUM_LAYERS) {
        fclose(f);
        return 0;   /* wrong format or architecture -> caller handles it */
    }
    int qmode = (int)hdr[1], ok = 1;
    ok &= quant_read_tensor(f, &Wemb[0][0], VOCAB, EMBED, qmode);
    for (int l = 0; l < NUM_LAYERS && ok; l++)
        for (int g = 0; g < NGATE && ok; g++) {
            ok &= quant_read_tensor(f, &Wg[l][g][0][0], HIDDEN, HIDDEN, qmode);
            ok &= quant_read_tensor(f, &Ug[l][g][0][0], HIDDEN, HIDDEN, qmode);
        }
    ok &= quant_read_tensor(f, &Why[0][0], VOCAB, HIDDEN, qmode);
    ok &= f32_read(f, &bg[0][0][0], NUM_LAYERS * NGATE * HIDDEN);
    ok &= f32_read(f, by, VOCAB);
    fclose(f);
    return ok;
}

/* ================================================================== */
/*  Quantization-aware training (QAT) — make a 1-bit / int8 model that   */
/*  is actually GOOD, not just small.                                    */
/*                                                                      */
/*  Plain quantization (above) rounds a finished float model and loses   */
/*  quality.  QAT instead trains the model to survive quantization: on   */
/*  every step the forward+backward pass runs on FAKE-QUANTIZED weights  */
/*  (so the model "sees" the low precision it will be deployed at), but  */
/*  Adam updates the full-precision LATENT weights.  Gradients flow      */
/*  through the quantizer via the straight-through estimator (treat the  */
/*  round/sign as identity in the backward pass).  Over training the     */
/*  latent weights move to values whose quantized form still works —     */
/*  this is the only way to get a genuinely strong 1-bit model.          */
/* ================================================================== */

/* In-place "fake quantize": round each row to the target precision and
 * immediately expand back to double (quantize->dequantize), so the forward
 * pass uses exactly the values the deployed model will. */
static void fake_quant_tensor(double *M, int rows, int cols, int qmode) {
    for (int i = 0; i < rows; i++) {
        double *row = M + (size_t)i * cols;
        if (qmode == 8) {
            double mx = 0.0;
            for (int j = 0; j < cols; j++) { double a = fabs(row[j]); if (a > mx) mx = a; }
            double scale = (mx > 0.0) ? mx / 127.0 : 1.0;
            for (int j = 0; j < cols; j++) {
                long v = lround(row[j] / scale);
                if (v >  127) v =  127;
                if (v < -127) v = -127;
                row[j] = (double)v * scale;
            }
        } else {
            double s = 0.0;
            for (int j = 0; j < cols; j++) s += fabs(row[j]);
            double alpha = (cols > 0 && s > 0.0) ? s / cols : 1.0;
            for (int j = 0; j < cols; j++) row[j] = (row[j] >= 0.0) ? alpha : -alpha;
        }
    }
}

/* Fake-quantize every weight matrix we export (biases stay full precision). */
static void fake_quant_all(int qmode) {
    fake_quant_tensor(&Wemb[0][0], VOCAB, EMBED, qmode);
    for (int l = 0; l < NUM_LAYERS; l++)
        for (int g = 0; g < NGATE; g++) {
            fake_quant_tensor(&Wg[l][g][0][0], HIDDEN, HIDDEN, qmode);
            fake_quant_tensor(&Ug[l][g][0][0], HIDDEN, HIDDEN, qmode);
        }
    fake_quant_tensor(&Why[0][0], VOCAB, HIDDEN, qmode);
}

/* One QAT step: back up latent weights, fake-quantize in place, run
 * forward+backward (grads only), restore latent, then Adam-update the latent
 * weights with the straight-through gradients. */
static double qat_step(const int *in, const int *tg, double hprev[NUM_LAYERS][HIDDEN],
                       int qmode, double *sWemb, double *sWg, double *sUg, double *sWhy) {
    memcpy(sWemb, Wemb, sizeof Wemb); memcpy(sWg, Wg, sizeof Wg);
    memcpy(sUg,   Ug,   sizeof Ug);   memcpy(sWhy, Why, sizeof Why);
    fake_quant_all(qmode);                       /* globals -> effective (quantized) */
    g_defer_adam = 1;
    double loss = train_step(in, tg, hprev);     /* forward+backward, no Adam */
    g_defer_adam = 0;
    memcpy(Wemb, sWemb, sizeof Wemb); memcpy(Wg, sWg, sizeof Wg);
    memcpy(Ug,   sUg,   sizeof Ug);   memcpy(Why, sWhy, sizeof Why);   /* restore latent */
    apply_adam();                                /* update latent via straight-through grads */
    return loss;
}

/* Train `iters` steps with QAT at the given precision.  Shadow (latent backup)
 * is heap-allocated, so it never counts against the 2 GB static-data limit. */
static void qat_train(const char *data, int data_len, int iters, int qmode) {
    if (data_len < SEQ_LEN + 1) { fprintf(stderr, "QAT: corpus too small\n"); return; }
    double *sWemb = malloc(sizeof Wemb), *sWg = malloc(sizeof Wg);
    double *sUg   = malloc(sizeof Ug),   *sWhy = malloc(sizeof Why);
    if (!sWemb || !sWg || !sUg || !sWhy) {
        fprintf(stderr, "QAT: out of memory for shadow weights\n");
        free(sWemb); free(sWg); free(sUg); free(sWhy); return;
    }
    double hprev[NUM_LAYERS][HIDDEN];
    memset(hprev, 0, sizeof hprev);
    int p = 0, inputs[SEQ_LEN], targets[SEQ_LEN];
    double smooth = -1.0;
    for (int it = 0; it < iters; it++) {
        if (p + SEQ_LEN + 1 >= data_len) { p = 0; memset(hprev, 0, sizeof hprev); }
        for (int k = 0; k < SEQ_LEN; k++) {
            inputs[k]  = (unsigned char)data[p + k]     & (VOCAB - 1);
            targets[k] = (unsigned char)data[p + k + 1] & (VOCAB - 1);
        }
        double loss = qat_step(inputs, targets, hprev, qmode, sWemb, sWg, sUg, sWhy);
        double per_char = loss / SEQ_LEN;
        smooth = (smooth < 0.0) ? per_char : 0.999 * smooth + 0.001 * per_char;
        p += SEQ_LEN;
        if (it % 200 == 0 || it == iters - 1)
            printf("  qat iter %5d/%d   loss/char = %.4f\n", it, iters, smooth);
        if (it > 0 && it % CKPT_EVERY == 0) {
            save_model(CKPT_PATH);          /* crash-safety: save latent weights */
            printf("  [auto-saved latent checkpoint at iter %d]\n", it);
        }
    }
    free(sWemb); free(sWg); free(sUg); free(sWhy);
}

/* ================================================================== */
/*  Optional online advisor — Sentinel can ask Claude Opus for safe,    */
/*  structural neural-net / security ideas, and (opt-in) learn from the  */
/*  answers.  This is the ONE part of Sentinel that talks to the cloud:  */
/*                                                                      */
/*    * OFF by default.  It does nothing unless ANTHROPIC_API_KEY is set */
/*      AND you explicitly run --ask-opus / --self-study.  The core      */
/*      stays 100% local.                                                */
/*    * The question is JSON-escaped in C and written to a temp file, so */
/*      it never passes through the shell (no injection).  The API key   */
/*      is read from the environment by curl, never by this program.     */
/*    * Opus output is only ever PRINTED or LEARNED FROM AS TEXT.  It is  */
/*      never executed as a command and never rewrites Sentinel's source.*/
/*      "Self-updating" here means the model updates its own WEIGHTS by   */
/*      studying the answer — not its code.                              */
/* ================================================================== */
#define OPUS_MODEL_DEFAULT "claude-opus-5-5"
#define OPUS_ENDPOINT      "https://api.anthropic.com/v1/messages"

/* Escape a string for embedding inside a JSON string literal. */
static void json_escape(const char *in, char *out, size_t outsz) {
    size_t o = 0;
    for (const unsigned char *p = (const unsigned char *)in; *p && o + 7 < outsz; p++) {
        unsigned char c = *p;
        switch (c) {
            case '"':  out[o++] = '\\'; out[o++] = '"';  break;
            case '\\': out[o++] = '\\'; out[o++] = '\\'; break;
            case '\n': out[o++] = '\\'; out[o++] = 'n';  break;
            case '\r': out[o++] = '\\'; out[o++] = 'r';  break;
            case '\t': out[o++] = '\\'; out[o++] = 't';  break;
            default:
                if (c < 0x20) o += (size_t)snprintf(out + o, outsz - o, "\\u%04x", c);
                else          out[o++] = (char)c;
        }
    }
    out[o] = '\0';
}

/* Pull the first "type":"text" block's text out of an API response, undoing
 * JSON string escapes.  Parsed in C (not the shell), so response content —
 * whatever it is — can never be executed. */
static int json_extract_text(const char *resp, char *out, size_t outsz) {
    const char *t = strstr(resp, "\"type\":\"text\"");
    const char *k = strstr(t ? t : resp, "\"text\":");
    if (!k) return 0;
    k = strchr(k + 7, '"');
    if (!k) return 0;
    k++;
    size_t o = 0;
    while (*k && o + 4 < outsz) {
        if (*k == '"') break;                      /* end of the JSON string */
        if (*k == '\\') {
            k++;
            switch (*k) {
                case 'n': out[o++] = '\n'; break;
                case 'r': out[o++] = '\r'; break;
                case 't': out[o++] = '\t'; break;
                case '"': out[o++] = '"';  break;
                case '\\': out[o++] = '\\'; break;
                case '/': out[o++] = '/';  break;
                case 'u': {
                    char hex[5] = { k[1], k[2], k[3], k[4], 0 };
                    long cp = (k[1] && k[2] && k[3] && k[4]) ? strtol(hex, NULL, 16) : 0;
                    if (cp < 0x80) out[o++] = (char)cp;
                    else if (cp < 0x800) { out[o++] = (char)(0xC0 | (cp >> 6)); out[o++] = (char)(0x80 | (cp & 0x3F)); }
                    else { out[o++] = (char)(0xE0 | (cp >> 12)); out[o++] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[o++] = (char)(0x80 | (cp & 0x3F)); }
                    if (k[1] && k[2] && k[3] && k[4]) k += 4;
                    break;
                }
                default: out[o++] = *k; break;
            }
            if (*k) k++;
        } else {
            out[o++] = *k++;
        }
    }
    out[o] = '\0';
    return 1;
}

/* Ask Claude Opus a question; fill `answer`.  Returns 0 on success, -1 on any
 * failure (no key, no network, no curl, empty response) with a message in
 * `answer`.  Opt-in: does nothing without ANTHROPIC_API_KEY. */
static int ask_opus(const char *question, const char *system_prompt,
                    char *answer, size_t ansz) {
    if (!getenv("ANTHROPIC_API_KEY")) {
        snprintf(answer, ansz,
                 "ANTHROPIC_API_KEY is not set. The online advisor is opt-in and off by "
                 "default — run `export ANTHROPIC_API_KEY=...` to enable it.");
        return -1;
    }
    const char *model = getenv("SENTINEL_OPUS_MODEL");
    if (!model || !*model) model = OPUS_MODEL_DEFAULT;

    char esc_q[8192], esc_s[2048], esc_m[128];
    json_escape(question, esc_q, sizeof esc_q);
    json_escape(system_prompt ? system_prompt : "", esc_s, sizeof esc_s);
    json_escape(model, esc_m, sizeof esc_m);

    char tmpl[] = "/tmp/sentinel_opus_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) { snprintf(answer, ansz, "could not create a temp request file"); return -1; }
    FILE *bf = fdopen(fd, "w");
    if (!bf) { close(fd); unlink(tmpl); snprintf(answer, ansz, "temp file error"); return -1; }
    fprintf(bf,
        "{\"model\":\"%s\",\"max_tokens\":1024,"
        "\"system\":\"%s\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"%s\"}]}",
        esc_m, esc_s, esc_q);
    fclose(bf);

    /* API key stays in the environment; only fixed strings + our temp path
     * reach the shell. The request body is @file, so the question is never
     * on the command line. */
    char cmd[4096];
    snprintf(cmd, sizeof cmd,
        "curl -s --max-time 60 %s "
        "-H 'content-type: application/json' "
        "-H \"x-api-key: $ANTHROPIC_API_KEY\" "
        "-H 'anthropic-version: 2023-06-01' "
        "--data-binary @%s",
        OPUS_ENDPOINT, tmpl);

    FILE *pp = popen(cmd, "r");
    if (!pp) { unlink(tmpl); snprintf(answer, ansz, "could not launch curl (is it installed?)"); return -1; }
    static char resp[1 << 16];
    size_t rn = fread(resp, 1, sizeof resp - 1, pp);
    resp[rn] = '\0';
    pclose(pp);
    unlink(tmpl);

    if (rn == 0) { snprintf(answer, ansz, "empty response (network blocked, or curl missing?)"); return -1; }
    if (!json_extract_text(resp, answer, ansz)) {
        const char *e = strstr(resp, "\"message\":");
        snprintf(answer, ansz, "no text block in response. raw: %.700s", e ? e : resp);
        return -1;
    }
    return 0;
}

/* Self-study loop: ask Opus a rotating set of safe neural-net / security
 * questions and LEARN from each answer (online training on the answer text),
 * checkpointing after each round.  The model updates its own weights — never
 * its code.  Opt-in; stops immediately if the advisor is unavailable. */
static int self_study(int rounds) {
    static const char *QBANK[] = {
        "In 3 short sentences, suggest one concrete, safe improvement to a character-level "
        "stacked-GRU language model trained with Adam in pure C. Be specific and practical.",
        "In 3 short sentences, explain gradient clipping and a good clip value for a small GRU, "
        "and why it helps training stability.",
        "In 3 short sentences, name one data-preprocessing or curriculum step that improves a "
        "small, security-focused character-level language model.",
        "In 3 short sentences, explain the tradeoff between widening (more hidden units) and "
        "deepening (more layers) a small GRU for a fixed parameter budget.",
        "In 3 short sentences, explain how int8 weight quantization keeps model quality while "
        "shrinking the model, and one pitfall to avoid.",
    };
    int nq = (int)(sizeof QBANK / sizeof QBANK[0]);
    const char *sys =
        "You are a concise neural-network engineering tutor helping a small, local, pure-C "
        "GRU language model understand how it can improve itself. Give short, practical, "
        "safe, legal guidance only — no exploit code, no attack instructions.";
    static char answer[1 << 16];
    int learned = 0;
    for (int r = 0; r < rounds; r++) {
        const char *q = QBANK[r % nq];
        printf("\n[self-study %d/%d] asking Opus:\n  Q: %s\n", r + 1, rounds, q);
        fflush(stdout);
        if (ask_opus(q, sys, answer, sizeof answer) != 0) {
            printf("  ! advisor unavailable: %s\n", answer);
            break;
        }
        printf("  A: %s\n", answer);
        size_t al = strlen(answer);
        if (al >= SEQ_LEN + 1) {
            int iters = (int)(al / SEQ_LEN) * 4;
            if (iters < 20) iters = 20;
            train(answer, (int)al, iters, 0);
            learned++;
            printf("  [self-update] studied the answer (%zu chars, %d iters) and checkpointed.\n",
                   al, iters);
        }
        save_model(CKPT_PATH);
    }
    printf("\n[self-study] updated the model from %d Opus answer(s); '%s' saved.\n",
           learned, CKPT_PATH);
    return learned;
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
 * Hard safety guard.  Returns 0 (unsafe) if the command string contains any
 * privilege-escalation or destructive pattern.  This is a best-effort denylist,
 * not a sandbox: the whole command is scanned (no truncation), and an
 * over-long command is refused outright rather than silently passed.
 */
static int is_command_safe(const char *cmd) {
    char low[4096];
    size_t n = strlen(cmd);
    if (n >= sizeof low) return 0;            /* refuse — never scan a partial cmd */
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)cmd[i]);
    low[n] = '\0';

    static const char *deny[] = {
        /* privilege escalation */
        "sudo", "su -", "su -l", "su root", "doas", "pkexec", "setpriv",
        "runuser", "/sudo", "\tsudo",
        /* destructive filesystem ops */
        "rm -rf /", "rm -rf /*", "rm -rf ~", "rm -fr /", "rm -r -f /",
        "rm -rf $home", "rm -rf .", "mkfs", "dd if=", "dd of=", "of=/dev/",
        "> /dev/sd", ">/dev/sd", "> /dev/sda", "mv / ",
        /* fork bombs / power state */
        ":(){", "fork()", "shutdown", "reboot", "halt", "poweroff",
        "init 0", "init 6", "systemctl ",
        /* perms / ownership */
        "chmod -r 777 /", "chmod 777 /", "chown -r",
        /* pipe-to-shell (the real RCE vector), reverse shells, persistence.
         * Note: a bare curl/wget is allowed (used for live CVE lookup); only
         * piping fetched content into a shell is blocked. */
        "|sh", "| sh", "|bash", "| bash", "curl|", "wget|", "|python", "| python",
        "nc -", "ncat", "socat", "/dev/tcp/", "mkfifo",
        ".ssh/", "authorized_keys", "bashrc", "crontab", "/etc/cron",
        /* credential files / firewall flush */
        "/etc/passwd", "/etc/shadow", "iptables -f", "ufw disable",
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

    /* Extract a sanitized CVE id (CVE-YYYY-NNNN..) from the task, if present;
     * only [0-9-] + the literal "CVE" reach the shell, so it's injection-safe. */
    char cve_id[24] = "";
    {
        const char *u = low;
        while ((u = strstr(u, "cve-")) != NULL) {
            int y = 0, d = 0; const char *q = u + 4;
            while (isdigit((unsigned char)q[y])) y++;
            if (y == 4 && q[4] == '-') {
                const char *r = q + 5;
                while (isdigit((unsigned char)r[d]) && d < 9) d++;
                if (d >= 4) {
                    snprintf(cve_id, sizeof cve_id, "CVE-%.4s-%.*s", q, d, r);
                    break;
                }
            }
            u += 4;
        }
    }

    /* Knowledge retrieval (RAG): grep the local corpus for CVE / malware /
     * OSINT facts.  This is how the SMALL LM stays small — knowledge lives in
     * the corpus and is retrieved, not baked into weights.  With SLM_ONLINE=1
     * and a CVE id in the query, it also fetches that CVE LIVE from the net. */
    if (strstr(low, "cve") || strstr(low, "vuln") || strstr(low, "malware") ||
        strstr(low, "ransomware") || strstr(low, "apt") || strstr(low, "ioc") ||
        strstr(low, "yara") || strstr(low, "trojan") || strstr(low, "osint") ||
        strstr(low, "recon") || strstr(low, "subdomain") || strstr(low, "breach") ||
        strstr(low, "threat")) {
        const char *topic = "knowledge";
        if (strstr(low,"cve")||strstr(low,"vuln"))                 topic = "cve";
        else if (strstr(low,"malware")||strstr(low,"ransomware")||
                 strstr(low,"apt")||strstr(low,"ioc")||
                 strstr(low,"yara")||strstr(low,"trojan"))         topic = "malware";
        else if (strstr(low,"osint")||strstr(low,"recon")||
                 strstr(low,"subdomain")||strstr(low,"breach"))    topic = "osint";
        char kw[256];
        extract_keywords(task, kw, sizeof kw);
        snprintf(cmd, cmdsz,
            /* live internet fetch of a specific CVE id (opt-in via SLM_ONLINE) */
            "if [ -n \"$SLM_ONLINE\" ] && [ -n '%s' ] && command -v curl >/dev/null; then "
            "  id='%s'; yr=$(echo \"$id\" | cut -d- -f2); "
            "  echo \"[online] fetching $id live from the internet...\"; "
            "  curl -s --max-time 12 \"https://raw.githubusercontent.com/trickest/cve/main/$yr/$id.md\""
            "  | sed -n '1,18p'; echo; fi; "
            /* local corpus retrieval (works fully offline) */
            "dir=corpus; [ -d \"$dir\" ] || dir=.; topic=%s; "
            "echo \"[retrieval agent] topic=$topic  searching $dir for: %s\"; "
            "sub=$(ls -d \"$dir\"/*\"$topic\"* 2>/dev/null | head -1); "
            "[ -n \"$sub\" ] && dir=\"$sub\"; "
            "hits=$(grep -rIl -i -E '%s' \"$dir\" --include='*.md' 2>/dev/null | head -3); "
            "[ -z \"$hits\" ] && hits=$(grep -rIl -i -E '%s' corpus 2>/dev/null --include='*.md' | head -3); "
            "if [ -z \"$hits\" ]; then "
            "echo 'no local match — run ./fetch_corpus.sh, or set SLM_ONLINE=1 with a CVE id'; "
            "else for f in $hits; do echo \"=== $f ===\"; "
            "grep -v -E '^!\\[|^[[:space:]]*$' \"$f\" 2>/dev/null | sed -n '1,8p'; "
            "echo; done; fi",
            cve_id, cve_id, topic, kw, kw, kw);
        return 0;  /* cyber specialism */
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
        /* Unique 0700 dir avoids the /tmp symlink-race of a fixed path. */
        char dir[64] = "/tmp/sentinel_agent_XXXXXX";
        char src[96], bin[96];
        if (!mkdtemp(dir)) {
            snprintf(cmd, cmdsz, "echo '[coding agent] could not create temp dir'");
            return kind;
        }
        snprintf(src, sizeof src, "%s/snippet.c", dir);
        snprintf(bin, sizeof bin, "%s/snippet", dir);
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
            /* note: 'rm -f' (not 'rm -rf') so the safety guard doesn't flag it */
            snprintf(cmd, cmdsz,
                "gcc %s -o %s 2>&1 && echo '[coding agent] compiled OK, running:' && %s; "
                "rm -f %s %s; rmdir %s 2>/dev/null",
                src, bin, bin, src, bin, dir);
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

/* ------------------------------------------------------------------ */
/*  Difficulty-tiered sub-agents.                                      */
/*                                                                     */
/*  Easy tasks get a small compute/parameter budget; hard tasks get a  */
/*  larger one.  A sub-agent re-execs into "--agent" mode, which never  */
/*  touches the big GRU weight arrays (demand-zero BSS stays            */
/*  unallocated), so each agent is only a few MB — that is why ~100 can */
/*  run at once.  The budget caps each agent's wall-clock compute so an */
/*  easy job can't hog resources.                                      */
/* ------------------------------------------------------------------ */
#define MAX_AGENTS 100

static const char *TIER_NAME[3]    = { "light", "medium", "heavy" };
static const int   TIER_BUDGET_K[3] = { 64, 512, 4096 };  /* notional param budget (x1000) */
static const int   TIER_TIMEOUT[3]  = { 3, 8, 20 };        /* compute cap, seconds          */

/* 0 = light, 1 = medium, 2 = heavy, from task length + keyword signals. */
static int estimate_difficulty(const char *task) {
    char low[512];
    size_t n = strlen(task);
    if (n >= sizeof low) n = sizeof low - 1;
    for (size_t i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)task[i]);
    low[n] = '\0';
    static const char *hard[] = {
        "compile","analyze","exploit","reverse","optimize","encrypt","decrypt",
        "vulnerability","algorithm","train","benchmark","fuzz","disassemble",
        "parser","recursion","simulate", NULL };
    static const char *med[] = {
        "write","build","scan","search","hash","cve","network","function",
        "script","convert","sort","port", NULL };
    for (int i = 0; hard[i]; i++) if (strstr(low, hard[i])) return 2;
    if (n > 80) return 2;
    for (int i = 0; med[i]; i++) if (strstr(low, med[i])) return 1;
    if (n > 40) return 1;
    return 0;
}

/* Runs as the spawned sub-agent at a given difficulty tier. */
static int run_agent(int tier, const char *task) {
    if (tier < 0 || tier > 2) tier = 1;
    int terse = getenv("SLM_AGENT_TERSE") != NULL;   /* compact output for big fan-outs */
    const char *kinds[] = { "CYBERSECURITY", "CODING", "GENERAL" };
    char cmd[2048];
    int kind = build_command(task, cmd, sizeof cmd);
    if (kind < 0 || kind > 2) kind = 2;

    if (!terse) {
        printf("\n  ┌─ sub-agent pid=%d  specialism=%s  tier=%s "
               "(param-budget≈%dK, %ds cap)\n",
               (int)getpid(), kinds[kind], TIER_NAME[tier],
               TIER_BUDGET_K[tier], TIER_TIMEOUT[tier]);
        printf("  │  task: %s\n", task);
    }

    if (getenv("SLM_NO_EXEC")) {
        if (terse) printf("  · agent pid=%d tier=%s PLAN-ONLY\n", (int)getpid(), TIER_NAME[tier]);
        else printf("  │  [plan-only mode: SLM_NO_EXEC set]\n  └─ would run: %s\n", cmd);
        return 0;
    }
    if (!is_command_safe(task) || !is_command_safe(cmd)) {
        if (terse) printf("  · agent pid=%d tier=%s REFUSED (unsafe)\n", (int)getpid(), TIER_NAME[tier]);
        else printf("  │  REFUSED by safety guard (sudo/destructive pattern).\n  └─ no action.\n");
        return 2;
    }

    /* difficulty-scaled compute cap via coreutils `timeout` (if present);
     * the validated command is passed through the environment to avoid any
     * shell-quoting issues. */
    setenv("SLM_AGENT_CMD", cmd, 1);
    char wrapped[160];
    snprintf(wrapped, sizeof wrapped,
        "command -v timeout >/dev/null && timeout %d sh -c 'eval \"$SLM_AGENT_CMD\"' "
        "|| sh -c 'eval \"$SLM_AGENT_CMD\"'", TIER_TIMEOUT[tier]);

    if (!terse) { printf("  │  exec: %s\n  │  ----- output -----\n", cmd); fflush(stdout); }

    FILE *pp = popen(wrapped, "r");
    if (!pp) { printf("  · agent pid=%d failed to launch shell\n", (int)getpid()); return 1; }
    char line[512];
    char first[512]; first[0] = '\0';
    while (fgets(line, sizeof line, pp)) {
        if (terse) { if (!first[0]) snprintf(first, sizeof first, "%s", line); }
        else printf("  │  %s", line);
    }
    int rc = pclose(pp);
    int code = (rc == -1) ? -1 : WEXITSTATUS(rc);
    if (terse) {
        size_t fl = strlen(first); if (fl && first[fl-1] == '\n') first[fl-1] = '\0';
        printf("  · agent pid=%d tier=%s %s%s(exit %d)\n", (int)getpid(),
               TIER_NAME[tier], code == 124 ? "[budget-capped] " : "",
               first[0] ? first : "", code);
    } else {
        if (code == 124) printf("  │  [compute budget exceeded — stopped at %ds cap]\n",
                                 TIER_TIMEOUT[tier]);
        printf("  └─ sub-agent pid=%d finished (exit %d)\n", (int)getpid(), code);
    }
    return 0;
}

/* Spawn ONE sub-agent process for `task` at `tier` and wait for it. */
static void spawn_agent_tier(const char *selfexe, const char *task, int tier) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return; }
    if (pid == 0) {
        char ts[8]; snprintf(ts, sizeof ts, "%d", tier);
        execl(selfexe, selfexe, "--agent", ts, task, (char *)NULL);
        perror("execl");
        _exit(127);
    }
    waitpid(pid, NULL, 0);
}

/* Fan out `count` sub-agents (capped at MAX_AGENTS) CONCURRENTLY for `task`,
 * each at `tier`, then reap them all.  Returns how many were launched. */
static int spawn_batch(const char *selfexe, const char *task, int count, int tier) {
    if (count < 1) count = 1;
    if (count > MAX_AGENTS) count = MAX_AGENTS;
    setenv("SLM_AGENT_TERSE", "1", 1);          /* compact per-agent output */
    pid_t pids[MAX_AGENTS];
    int launched = 0;
    for (int i = 0; i < count; i++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); break; }
        if (pid == 0) {
            char ts[8]; snprintf(ts, sizeof ts, "%d", tier);
            execl(selfexe, selfexe, "--agent", ts, task, (char *)NULL);
            _exit(127);
        }
        pids[launched++] = pid;
    }
    for (int i = 0; i < launched; i++) waitpid(pids[i], NULL, 0);
    unsetenv("SLM_AGENT_TERSE");
    return launched;
}

/* Run a task and capture the agent output into `out` (for the GUI/TUI).
 * Honors the safety guard and SLM_NO_EXEC. Returns bytes written. */
static size_t run_task_capture(const char *task, char *out, size_t outsz) {
    int tier = estimate_difficulty(task);
    char cmd[2048];
    int kind = build_command(task, cmd, sizeof cmd);
    if (kind < 0 || kind > 2) kind = 2;
    const char *kn[] = { "cybersecurity", "coding", "general" };
    size_t off = 0;
    off += snprintf(out + off, outsz - off,
                    "[%s agent · tier %s]\n", kn[kind], TIER_NAME[tier]);
    if (!is_command_safe(task) || !is_command_safe(cmd)) {
        off += snprintf(out + off, outsz - off,
                        "REFUSED by safety guard (sudo/destructive pattern).\n");
        return off;
    }
    if (getenv("SLM_NO_EXEC")) {
        off += snprintf(out + off, outsz - off, "[plan-only] would run:\n%s\n", cmd);
        return off;
    }
    setenv("SLM_AGENT_CMD", cmd, 1);
    char wrapped[160];
    snprintf(wrapped, sizeof wrapped,
        "command -v timeout >/dev/null && timeout %d sh -c 'eval \"$SLM_AGENT_CMD\"' "
        "|| sh -c 'eval \"$SLM_AGENT_CMD\"'", TIER_TIMEOUT[tier]);
    FILE *pp = popen(wrapped, "r");
    if (!pp) { off += snprintf(out + off, outsz - off, "failed to launch shell\n"); return off; }
    char line[512];
    while (off < outsz - 1 && fgets(line, sizeof line, pp))
        off += snprintf(out + off, outsz - off, "%s", line);
    pclose(pp);
    return off;
}

/* ================================================================== */
/*  Web GUI — a tiny HTTP server in pure C (POSIX sockets, no deps).    */
/*                                                                     */
/*  Binds to 127.0.0.1 ONLY (localhost): the agent runs shell commands, */
/*  so it must not be exposed to the network.  Serves a chat page and a */
/*  /api endpoint that runs a task and returns the output.             */
/* ================================================================== */
static const char *GUI_HTML =
"<!doctype html><html><head><meta charset=utf-8><title>Sentinel</title>"
"<meta name=viewport content='width=device-width,initial-scale=1'>"
"<style>body{background:#0a0f1f;color:#e6f6ff;font-family:ui-monospace,Menlo,Consolas,monospace;margin:0}"
"header{padding:14px 18px;border-bottom:1px solid #18324a;color:#22d3ee;font-weight:700;font-size:20px}"
"#log{padding:16px 18px;height:calc(100vh - 150px);overflow:auto;white-space:pre-wrap;font-size:13px}"
".u{color:#34d399}.a{color:#cfe8ff}.s{color:#5b7a99}"
"form{display:flex;gap:8px;padding:12px 18px;border-top:1px solid #18324a}"
"input{flex:1;background:#0b1426;border:1px solid #18324a;color:#e6f6ff;padding:10px;border-radius:8px;font:inherit}"
"button{background:#22d3ee;border:0;color:#04121c;padding:10px 16px;border-radius:8px;font-weight:700;cursor:pointer}"
"</style></head><body>"
"<header>\xe2\x97\x86 Sentinel \xc2\xb7 local AI agent</header>"
"<div id=log><span class=s>Type a task (e.g. \"list files\", \"show me a cve about android\", "
"\"write a C function to reverse a string\"). Runs locally on your machine.</span>\n\n</div>"
"<form onsubmit=\"send();return false\"><input id=q autofocus placeholder='ask Sentinel...'>"
"<button>Send</button></form>"
"<script>"
"var log=document.getElementById('q');var L=document.getElementById('log');"
"function add(c,t){var s=document.createElement('span');s.className=c;s.textContent=t;L.appendChild(s);L.scrollTop=L.scrollHeight;}"
"async function send(){var q=document.getElementById('q');var t=q.value.trim();if(!t)return;q.value='';"
"add('u','\\u25b8 '+t+'\\n');add('s','running...\\n');"
"try{var r=await fetch('/api',{method:'POST',body:t});var x=await r.text();"
"L.lastChild.remove();add('a',x+'\\n\\n');}catch(e){L.lastChild.remove();add('s','error: '+e+'\\n');}}"
"</script></body></html>";

static volatile sig_atomic_t g_serve_stop = 0;
static void on_sigint(int s) { (void)s; g_serve_stop = 1; }

static void http_send(int c, const char *status, const char *ctype, const char *body, size_t blen) {
    char hdr[256];
    int hl = snprintf(hdr, sizeof hdr,
        "HTTP/1.0 %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\n"
        "Connection: close\r\n\r\n", status, ctype, blen);
    if (write(c, hdr, hl) < 0) return;
    size_t off = 0;
    while (off < blen) { ssize_t w = write(c, body + off, blen - off); if (w <= 0) break; off += w; }
}

static int serve_http(int port) {
    signal(SIGINT, on_sigint);
    signal(SIGPIPE, SIG_IGN);
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); return 1; }
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   /* localhost ONLY */
    a.sin_port = htons((unsigned short)port);
    if (bind(s, (struct sockaddr *)&a, sizeof a) < 0) { perror("bind"); close(s); return 1; }
    if (listen(s, 16) < 0) { perror("listen"); close(s); return 1; }
    printf("Sentinel GUI on http://127.0.0.1:%d  (localhost only — Ctrl-C to stop)\n", port);
    fflush(stdout);

    static char req[8192], out[1 << 16];
    while (!g_serve_stop) {
        int c = accept(s, NULL, NULL);
        if (c < 0) { if (g_serve_stop) break; continue; }
        ssize_t n = read(c, req, sizeof req - 1);
        if (n <= 0) { close(c); continue; }
        req[n] = '\0';
        if (strncmp(req, "POST /api", 9) == 0) {
            char *body = strstr(req, "\r\n\r\n");
            body = body ? body + 4 : (char *)"";
            size_t blen = run_task_capture(body, out, sizeof out);
            http_send(c, "200 OK", "text/plain; charset=utf-8", out, blen);
        } else {
            http_send(c, "200 OK", "text/html; charset=utf-8", GUI_HTML, strlen(GUI_HTML));
        }
        close(c);
    }
    close(s);
    printf("\nGUI stopped.\n");
    return 0;
}

/* ================================================================== */
/*  TUI — an ANSI-styled interactive terminal interface (no ncurses).  */
/* ================================================================== */
static void run_tui(const char *selfexe, int params) {
    printf("\033[2J\033[H");                       /* clear screen, home */
    printf("\033[36m");
    printf("  ╔══════════════════════════════════════════════════════════╗\n");
    printf("  ║   ◆  S E N T I N E L   ·   local deep-GRU AI agent        ║\n");
    printf("  ╚══════════════════════════════════════════════════════════╝\033[0m\n");
    printf("  \033[90m%d params · Adam · fork-spawned sub-agents · 100%% local\033[0m\n\n", params);
    printf("  \033[90mTASK: <x> | FANOUT: <n> <x> | \"show me a cve about <x>\" | /quit\033[0m\n\n");
    fflush(stdout);

    char line[4096];
    int spawned = 0;
    for (;;) {
        printf("\033[32m sentinel \033[36m\xe2\x9d\xaf\033[0m ");   /* green/cyan prompt */
        fflush(stdout);
        if (!fgets(line, sizeof line, stdin)) break;
        size_t len = strlen(line);
        while (len && (line[len-1] == '\n' || line[len-1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;

        char low[256]; size_t ln = len < sizeof low - 1 ? len : sizeof low - 1;
        for (size_t i = 0; i < ln; i++) low[i] = (char)tolower((unsigned char)line[i]);
        low[ln] = '\0';
        char *fan = strstr(line, "FANOUT:");
        char *trig = strstr(line, "TASK:");
        int cve = (!fan && !trig) && (strstr(low, "cve") || strstr(low, "vuln"));

        printf("\033[90m  ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄\033[0m\n");
        if (fan) {
            char *p = fan + 7;
            while (*p == ' ') p++;
            int count = atoi(p);
            while (*p && *p != ' ') p++;
            while (*p == ' ') p++;
            int tier = estimate_difficulty(p);
            printf("  \033[36mfan-out %d agents (tier %s)\033[0m\n", count, TIER_NAME[tier]);
            spawned += spawn_batch(selfexe, p, count, tier);
        } else {
            char *task = trig ? trig + 5 : line; while (*task == ' ') task++;
            int tier = estimate_difficulty(task);
            printf("  \033[36m▸ %s  \033[90m(tier %s)\033[0m\n", task, TIER_NAME[tier]);
            spawn_agent_tier(selfexe, task, tier);
            spawned++;
            (void)cve;
        }
        printf("\n");
    }
    printf("\n  \033[90m%d sub-agents this session. bye.\033[0m\n", spawned);
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
"Sentinel v%s — a self-contained deep-GRU AI agent in pure C.\n"
"A character-level neural net (no libraries) that learns from text, answers\n"
"CVE questions from its corpus, and forks sub-agents to do small tasks.\n"
"\n"
"USAGE\n"
"  %s                      train on the built-in/checkpointed model, print a\n"
"                            sample, then enter the interactive AI loop\n"
"  %s --train <path...>    consolidate files/dirs into a corpus and train on it\n"
"  %s --agent \"<task>\"     run once as a single sub-agent for <task>\n"
"  %s --tui                styled terminal UI (interactive agent console)\n"
"  %s --serve [port]       web GUI on http://127.0.0.1:8080 (localhost only)\n"
"  %s --quantize <int8|1bit> [out]\n"
"                            export a SMALL quantized copy of the model for\n"
"                            shipping (int8 ~8x / 1bit ~64x smaller); the float\n"
"                            checkpoint is never touched\n"
"  %s --sample-quant <file> [n]\n"
"                            load a quantized model and print an n-char sample\n"
"                            (verify the small model still generates)\n"
"  %s --quantize-train <int8|1bit> [corpus...]\n"
"                            quantization-AWARE training: train a model that is\n"
"                            actually GOOD at low precision (straight-through\n"
"                            estimator). Writes a robust float ckpt + deployable\n"
"                            quantized model. Iters via SLM_EPOCHS.\n"
"  %s --ask-opus \"<question>\"\n"
"                            OPT-IN online advisor: ask Claude Opus one safe\n"
"                            neural-net/security question (needs ANTHROPIC_API_KEY)\n"
"  %s --self-study [rounds]\n"
"                            OPT-IN: the model asks Opus for ideas and LEARNS from\n"
"                            the answers (updates its weights, never its code)\n"
"  %s --generate [n] [seed]  print n chars from the checkpoint (no training)\n"
"  %s --gradcheck          numerically verify backprop (build tiny first)\n"
"  %s --help | -h          show this help\n"
"  %s --version            print version\n"
"\n"
"INTERACTIVE LOOP (what to type once it's running)\n"
"  TASK: <something>         spawn ONE sub-agent (cyber / coding / general):\n"
"      TASK: list files in this directory\n"
"      TASK: write a C function to reverse a string\n"
"      TASK: scan local network ports\n"
"  FANOUT: <n> <something>   spawn up to 100 sub-agents CONCURRENTLY for a task,\n"
"                            e.g.  FANOUT: 100 scan local network ports\n"
"  cve / malware / osint <x> retrieval (RAG): greps the local corpus for real\n"
"                            CVE / malware-analysis / OSINT facts (no 'TASK:' needed).\n"
"                            Mention a CVE-id (e.g. CVE-2021-44228) with SLM_ONLINE=1\n"
"                            to fetch it LIVE from the internet.\n"
"  (each agent is auto-assigned a difficulty tier — light/medium/heavy — which\n"
"   sets its compute/parameter budget; easy tasks get a smaller budget.)\n"
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
"  SLM_LR=F                  override the base learning rate (default 0.002)\n"
"  SLM_NO_EXEC=1             agents PLAN ONLY — print commands, run nothing\n"
"  ANTHROPIC_API_KEY=...     enables the OPT-IN online advisor (--ask-opus /\n"
"                            --self-study). Unset = the core is 100%% local.\n"
"  SENTINEL_OPUS_MODEL=...   advisor model id (default claude-opus-5-5)\n"
"\n"
"SAFETY\n"
"  Sub-agents run real shell commands but a hard guard refuses 'sudo' and\n"
"  destructive patterns (rm -rf /, mkfs, dd, fork bombs, shutdown, ...).\n"
"  Set SLM_NO_EXEC=1 to disable command execution entirely.\n",
        VERSION, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog, prog);
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

    /* ---- optional online advisor: ask Claude Opus one question (opt-in) ---- */
    if (argc >= 3 && strcmp(argv[1], "--ask-opus") == 0) {
        static char ans[1 << 16];
        const char *sys =
            "You are a concise neural-network and security engineering advisor for a small, "
            "local, pure-C GRU model. Give short, practical, safe, legal guidance only.";
        int rc = ask_opus(argv[2], sys, ans, sizeof ans);
        printf("%s\n", ans);
        return rc == 0 ? 0 : 1;
    }

    /* ---- sub-agent mode:  --agent <tier> <task>  (legacy:  --agent <task>) ---- */
    if (argc >= 4 && strcmp(argv[1], "--agent") == 0)
        return run_agent(atoi(argv[2]), argv[3]);
    if (argc == 3 && strcmp(argv[1], "--agent") == 0)
        return run_agent(1, argv[2]);

    /* ---- GUI / TUI agent consoles (don't allocate the big LM weights) ---- */
    if (argc >= 2 && strcmp(argv[1], "--serve") == 0)
        return serve_http(argc >= 3 ? atoi(argv[2]) : 8080);
    if (argc >= 2 && strcmp(argv[1], "--tui") == 0) {
        char selfexe[PATH_MAX]; self_path(selfexe, sizeof selfexe, argv[0]);
        int params = (int)(VOCAB*EMBED + NUM_LAYERS*NGATE*(2*HIDDEN*HIDDEN + HIDDEN)
                           + VOCAB*HIDDEN + VOCAB);
        run_tui(selfexe, params);
        return 0;
    }

    srand(1234567u);
    init_weights();

    if (argc >= 2 && strcmp(argv[1], "--gradcheck") == 0) { gradcheck(); return 0; }

    /* ---- quantized export (non-destructive) and quantized-sample verify ---- */
    if (argc >= 3 && strcmp(argv[1], "--quantize") == 0) {
        int qmode = (strcmp(argv[2], "1bit") == 0 || strcmp(argv[2], "1") == 0) ? 1 : 8;
        const char *out = (argc >= 4) ? argv[3]
                        : (qmode == 1 ? "sentinel-1bit.bin" : "sentinel-int8.bin");
        if (load_model(CKPT_PATH))
            printf("Loaded float checkpoint '%s'.\n", CKPT_PATH);
        else
            printf("No float checkpoint '%s' — quantizing current (untrained) weights.\n",
                   CKPT_PATH);
        if (!save_quant(out, qmode)) {
            fprintf(stderr, "quantize: failed to write '%s'\n", out);
            return 1;
        }
        struct stat sf, sq;
        long params = (long)(VOCAB*EMBED + NUM_LAYERS*NGATE*(2*HIDDEN*HIDDEN + HIDDEN)
                             + VOCAB*HIDDEN + VOCAB);
        printf("Quantized to %s and wrote '%s'.\n", qmode == 1 ? "1-bit" : "int8", out);
        printf("  params: %ld   (~%.2f bytes/param in this file)\n", params,
               stat(out, &sq) == 0 ? (double)sq.st_size / (double)params : 0.0);
        if (stat(CKPT_PATH, &sf) == 0 && stat(out, &sq) == 0 && sq.st_size > 0)
            printf("  float checkpoint: %.1f MB  ->  quantized: %.1f MB  (%.1fx smaller)\n",
                   sf.st_size / 1e6, sq.st_size / 1e6, (double)sf.st_size / (double)sq.st_size);
        printf("  your float checkpoint '%s' is untouched.\n", CKPT_PATH);
        return 0;
    }
    if (argc >= 3 && strcmp(argv[1], "--sample-quant") == 0) {
        if (!load_quant(argv[2])) {
            fprintf(stderr, "cannot load quantized model '%s' (wrong format or arch)\n", argv[2]);
            return 1;
        }
        int n = (argc >= 4) ? atoi(argv[3]) : 200;
        if (n <= 0) n = 200;
        printf("Loaded quantized model '%s'. Sample (seeded 't'):\n  \"", argv[2]);
        sample('t', n);
        printf("\"\n");
        return 0;
    }

    /* ---- pure inference: generate n chars from the checkpoint, no training ---- */
    if (argc >= 2 && strcmp(argv[1], "--generate") == 0) {
        if (!load_model(CKPT_PATH))
            fprintf(stderr, "no checkpoint '%s' — generating from untrained weights\n", CKPT_PATH);
        int n = (argc >= 3) ? atoi(argv[2]) : 400;
        if (n <= 0) n = 400;
        int seed = (argc >= 4 && argv[3][0]) ? (argv[3][0] & (VOCAB - 1)) : 't';
        sample(seed, n);          /* prints n chars + newline to stdout */
        return 0;
    }

    int loaded = load_model(CKPT_PATH);
    if (loaded) printf("Loaded checkpoint '%s' — continuing to learn.\n", CKPT_PATH);
    else        printf("No checkpoint — starting from fixed static weights.\n");

    int iters = 2000;
    const char *env_ep = getenv("SLM_EPOCHS");
    if (env_ep) { int v = atoi(env_ep); if (v > 0) iters = v; }
    const char *env_lr = getenv("SLM_LR");
    if (env_lr) { double v = atof(env_lr); if (v > 0.0) g_lr = v; }

    /* ---- self-study: learn neural-net/security ideas from Opus (opt-in) ---- */
    if (argc >= 2 && strcmp(argv[1], "--self-study") == 0) {
        int rounds = (argc >= 3) ? atoi(argv[2]) : 3;
        if (rounds < 1) rounds = 1;
        if (rounds > 50) rounds = 50;
        printf("=== self-study: the model asks Opus for ideas and learns from the answers ===\n");
        printf("(opt-in online mode; the local core otherwise never phones home)\n");
        self_study(rounds);
        return 0;
    }

    /* ---- quantization-aware training: make a GOOD 1-bit / int8 model ---- */
    if (argc >= 3 && strcmp(argv[1], "--quantize-train") == 0) {
        int qmode = (strcmp(argv[2], "1bit") == 0 || strcmp(argv[2], "1") == 0) ? 1 : 8;
        const char *data = CORPUS;
        int dlen = (int)(sizeof(CORPUS) - 1);
        Corpus c = {0};
        if (argc >= 4) {                       /* optional corpus path(s) to QAT on */
            printf("Consolidating QAT corpus from %d path(s):\n", argc - 3);
            for (int i = 3; i < argc; i++) corpus_append_path(&c, argv[i]);
            if (c.len < (size_t)(SEQ_LEN + 1)) {
                fprintf(stderr, "QAT corpus too small — using the embedded corpus.\n");
            } else { c.buf[c.len] = '\0'; data = c.buf; dlen = (int)c.len; }
        }
        printf("=== quantization-aware training (%s) for %d iters ===\n",
               qmode == 1 ? "1-bit" : "int8", iters);
        qat_train(data, dlen, iters, qmode);
        free(c.buf);
        save_model(CKPT_PATH);                 /* latent float ckpt (quantization-robust) */
        const char *out = (qmode == 1) ? "sentinel-1bit-qat.bin" : "sentinel-int8-qat.bin";
        if (save_quant(out, qmode))
            printf("Saved quantization-robust float checkpoint '%s' + deployable %s model '%s'.\n",
                   CKPT_PATH, qmode == 1 ? "1-bit" : "int8", out);
        if (load_quant(out)) {                 /* prove the deployable model generates well */
            printf("Sample from the %s QAT model (seeded 't'):\n  \"",
                   qmode == 1 ? "1-bit" : "int8");
            sample('t', 160);
            printf("\"\n");
        }
        return 0;
    }

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
    printf("=== Sentinel : self-contained deep GRU network ===\n");
    printf("arch: VOCAB=%d EMBED=%d HIDDEN=%d LAYERS=%d SEQ_LEN=%d  (%d params)\n",
           VOCAB, EMBED, HIDDEN, NUM_LAYERS, SEQ_LEN,
           (int)(VOCAB*EMBED + NUM_LAYERS*NGATE*(2*HIDDEN*HIDDEN + HIDDEN)
                 + VOCAB*HIDDEN + VOCAB));
    printf("Training loop (BPTT through %d stacked GRU layers):\n", NUM_LAYERS);
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

        char *fan = strstr(line, "FANOUT:");
        char *trig = strstr(line, "TASK:");
        /* also treat a plain CVE/vuln question as a lookup, no "TASK:" needed */
        char low[256]; size_t ln = len < sizeof low - 1 ? len : sizeof low - 1;
        for (size_t i = 0; i < ln; i++) low[i] = (char)tolower((unsigned char)line[i]);
        low[ln] = '\0';
        int is_cve_q = (!trig && !fan) && (strstr(low, "cve") || strstr(low, "vuln"));

        if (fan) {
            /* FANOUT: <count> <task>  — spawn up to 100 concurrent sub-agents */
            char *p = fan + 7;
            while (*p == ' ') p++;
            int count = atoi(p);
            while (*p && *p != ' ') p++;     /* skip the number */
            while (*p == ' ') p++;
            if (count < 1) count = 1;
            int tier = estimate_difficulty(p);
            printf("[fan-out] launching %d concurrent sub-agents (tier=%s) for: \"%s\"\n",
                   count > MAX_AGENTS ? MAX_AGENTS : count, TIER_NAME[tier], p);
            fflush(stdout);
            int n = spawn_batch(selfexe, p, count, tier);
            printf("[fan-out] %d sub-agents completed.\n", n);
            spawned += n;
        } else if (trig || is_cve_q) {
            char *task = trig ? trig + 5 : line;
            while (*task == ' ') task++;
            int tier = estimate_difficulty(task);
            printf("[trigger detected] spawning sub-agent (tier=%s) for: \"%s\"\n",
                   TIER_NAME[tier], task);
            fflush(stdout);
            spawn_agent_tier(selfexe, task, tier);
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
