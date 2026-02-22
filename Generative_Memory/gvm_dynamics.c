#include "gvm_dynamics.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define OVERLAY_INITIAL_SIZE 1024
#define OVERLAY_LOAD_FACTOR 0.7

// --- INTERNAL HELPERS ---

static uint64_t gmem_hash(uint64_t addr) {
  uint64_t h = 14695981039346656037ULL;
  h ^= addr;
  h *= 1099511628211ULL;
  return h;
}

static void gmem_overlay_insert(gmem_ctx_t ctx, uint64_t addr, float value) {
  if (ctx->overlay_count >= ctx->overlay_size * OVERLAY_LOAD_FACTOR) {
    size_t old_size = ctx->overlay_size;
    gmem_entry_t *old_entries = ctx->overlay;

    ctx->overlay_size *= 2;
    ctx->overlay =
        (gmem_entry_t *)calloc(ctx->overlay_size, sizeof(gmem_entry_t));
    if (!ctx->overlay)
      exit(1);
    ctx->overlay_count = 0;

    for (size_t i = 0; i < old_size; i++) {
      if (old_entries[i].active) {
        uint64_t h = gmem_hash(old_entries[i].virtual_addr);
        size_t idx = (size_t)(h % ctx->overlay_size);
        while (ctx->overlay[idx].active)
          idx = (idx + 1) % ctx->overlay_size;
        ctx->overlay[idx].virtual_addr = old_entries[i].virtual_addr;
        ctx->overlay[idx].value = old_entries[i].value;
        ctx->overlay[idx].active = 1;
        ctx->overlay_count++;
      }
    }
    free(old_entries);
  }

  uint64_t h = gmem_hash(addr);
  size_t idx = (size_t)(h % ctx->overlay_size);
  while (ctx->overlay[idx].active) {
    if (ctx->overlay[idx].virtual_addr == addr) {
      ctx->overlay[idx].value = value;
      return;
    }
    idx = (idx + 1) % ctx->overlay_size;
  }
  ctx->overlay[idx].virtual_addr = addr;
  ctx->overlay[idx].value = value;
  ctx->overlay[idx].active = 1;
  ctx->overlay_count++;
}

static int gmem_overlay_lookup(gmem_ctx_t ctx, uint64_t addr, float *out_val) {
  if (ctx->overlay_size == 0)
    return 0;
  uint64_t h = gmem_hash(addr);
  size_t idx = (size_t)(h % ctx->overlay_size);
  size_t start_idx = idx;
  while (ctx->overlay[idx].active) {
    if (ctx->overlay[idx].virtual_addr == addr) {
      *out_val = ctx->overlay[idx].value;
      return 1;
    }
    idx = (idx + 1) % ctx->overlay_size;
    if (idx == start_idx)
      break;
  }
  return 0;
}

// --- HILBERT ENCODING ---

uint64_t gmem_hilbert_encode(uint64_t n, uint64_t x, uint64_t y) {
  uint64_t d = 0;
  for (uint64_t s = n / 2; s > 0; s /= 2) {
    uint64_t rx = (x & s) > 0;
    uint64_t ry = (y & s) > 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry == 0) {
      if (rx == 1) {
        x = n - 1 - x;
        y = n - 1 - y;
      }
      uint64_t t = x;
      x = y;
      y = t;
    }
  }
  return d;
}

void gmem_bulk_hilbert_encode(uint64_t n, const uint64_t *x, const uint64_t *y,
                              uint64_t *h_out, uint64_t count) {
  if (!x || !y || !h_out)
    return;
  for (uint64_t i = 0; i < count; i++) {
    h_out[i] = gmem_hilbert_encode(n, x[i], y[i]);
  }
}

// --- VRNS SYNTHESIS ---

static const uint16_t MODULI[8] = {251, 257, 263, 269, 271, 277, 281, 283};

float gmem_fetch_f32(gmem_ctx_t ctx, uint64_t virtual_addr) {
  float cached;
  if (gmem_overlay_lookup(ctx, virtual_addr, &cached))
    return cached;

  uint64_t x = virtual_addr ^ ctx->seed;
  double accumulator = 0.0;
  for (int i = 0; i < 8; i++) {
    accumulator += (double)(x % MODULI[i]) / (double)MODULI[i];
  }
  return (float)(accumulator - (long)accumulator);
}

void gmem_write_f32(gmem_ctx_t ctx, uint64_t virtual_addr, float value) {
  gmem_overlay_insert(ctx, virtual_addr, value);
}

// --- TRINITY CRT SOLVE ---

static __int128 mod_inverse(__int128 a, __int128 m) {
  __int128 m0 = m, t, q;
  __int128 x0 = 0, x1 = 1;
  if (m == 1)
    return 0;
  while (a > 1) {
    q = a / m;
    t = m;
    m = a % m, a = t;
    t = x0;
    x0 = x1 - q * x0;
    x1 = t;
  }
  if (x1 < 0)
    x1 += m0;
  return x1;
}

uint64_t gmem_trinity_solve_rns(gmem_ctx_t ctx, uint64_t x, uint64_t y,
                                uint64_t intention_sig, uint64_t law_sig,
                                uint64_t event_sig) {
  uint64_t result_sig = intention_sig ^ law_sig ^ event_sig ^ ctx->seed;
  const uint64_t m[] = {GMEM_MOD_GOLDILOCKS, GMEM_MOD_SAFE, GMEM_MOD_ULTRA};
  unsigned __int128 residues[3];

  for (int i = 0; i < 3; i++) {
    uint64_t mix = result_sig;
    mix ^= (x ^ (y << 13));
    mix *= m[i];
    residues[i] = mix % m[i];
  }

  unsigned __int128 m_prod = (unsigned __int128)m[0] * m[1] * m[2];
  unsigned __int128 sum = 0;
  for (int i = 0; i < 3; i++) {
    unsigned __int128 Mi = m_prod / m[i];
    unsigned __int128 yi =
        (unsigned __int128)mod_inverse((__int128)Mi, (__int128)m[i]);
    sum = (sum + residues[i] * Mi * yi) % m_prod;
  }

  return (uint64_t)(sum & 0xFFFFFFFFFFFFFFFFULL);
}

// --- PUBLIC API ---

gmem_ctx_t gmem_create(uint64_t seed) {
  gmem_ctx_t ctx = (gmem_ctx_t)calloc(1, sizeof(struct gmem_context));
  if (ctx) {
    ctx->seed = seed;
    ctx->overlay_size = OVERLAY_INITIAL_SIZE;
    ctx->overlay =
        (gmem_entry_t *)calloc(ctx->overlay_size, sizeof(gmem_entry_t));
    if (!ctx->overlay) {
      free(ctx);
      return NULL;
    }
  }
  return ctx;
}

void gmem_destroy(gmem_ctx_t ctx) {
  if (ctx) {
    if (ctx->overlay)
      free(ctx->overlay);
    free(ctx);
  }
}

void gmem_bulk_fetch_f32(gmem_ctx_t ctx, const uint64_t *h_indices,
                         float *buffer, size_t count) {
  if (!ctx || !h_indices || !buffer)
    return;
  for (size_t i = 0; i < count; i++) {
    buffer[i] = gmem_fetch_f32(ctx, h_indices[i]);
  }
}
