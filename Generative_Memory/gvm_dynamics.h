#ifndef GVM_DYNAMICS_H
#define GVM_DYNAMICS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// --- CORE TYPES ---

typedef struct gmem_context *gmem_ctx_t;

typedef struct {
  uint64_t virtual_addr;
  float value;
  int active;
} gmem_entry_t;

struct gmem_context {
  uint64_t seed;
  gmem_entry_t *overlay;
  size_t overlay_size;
  size_t overlay_count;
};

// --- CORE API ---

gmem_ctx_t gmem_create(uint64_t seed);
void gmem_destroy(gmem_ctx_t ctx);

float gmem_fetch_f32(gmem_ctx_t ctx, uint64_t virtual_addr);
void gmem_write_f32(gmem_ctx_t ctx, uint64_t virtual_addr, float value);

// --- HILBERT MAPPING ---

uint64_t gmem_hilbert_encode(uint64_t n, uint64_t x, uint64_t y);
void gmem_bulk_hilbert_encode(uint64_t n, const uint64_t *x, const uint64_t *y,
                              uint64_t *h_out, uint64_t count);

// --- TRINITY CRT SOLVE ---

#define GMEM_MOD_GOLDILOCKS 0xFFFFFFFF00000001ULL
#define GMEM_MOD_SAFE 0xFFFFFFFF7FFFFFFFULL
#define GMEM_MOD_ULTRA 0xFFFFFFFFFFFFFFC5ULL

// Note: Using __int128 for CRT intermediate calculations
uint64_t gmem_trinity_solve_rns(gmem_ctx_t ctx, uint64_t x, uint64_t y,
                                uint64_t intention_sig, uint64_t law_sig,
                                uint64_t event_sig);

// --- BULK FETCH ---

void gmem_bulk_fetch_f32(gmem_ctx_t ctx, const uint64_t *h_indices,
                         float *buffer, size_t count);

#ifdef __cplusplus
}
#endif

#endif // GVM_DYNAMICS_H
