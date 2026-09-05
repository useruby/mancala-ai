/* Repository-owned reader for the isolated canonical KVTB1 diagnostic artifact. */
#ifndef CANONICAL_KVTB_H
#define CANONICAL_KVTB_H
#include <stdint.h>
#include "rules.h"
int canonical_tablebase_load(const char *path);
void canonical_tablebase_close(void);
int canonical_tablebase_known(int active_stones);
int canonical_tablebase_value(const position *p);
long long canonical_tablebase_hits(void);
void canonical_tablebase_reset_request_metrics(void);
long long canonical_tablebase_request_lookups(void);
long long canonical_tablebase_request_hits(void);
int canonical_tablebase_first_hit_depth(void);
int canonical_tablebase_min_hit_tier(void);
int canonical_tablebase_max_hit_tier(void);
long long canonical_tablebase_tier_hits(int tier);
void canonical_tablebase_record_lookup(int active_stones, int depth);
typedef struct {
  int active_stones;
  uint64_t rank;
  uint64_t offset;
  int raw_value;
  int store_margin;
  int upstream_value;
  int player_zero_value;
} canonical_tablebase_info;
int canonical_tablebase_diagnostics(const position *p, canonical_tablebase_info *info);
#endif
