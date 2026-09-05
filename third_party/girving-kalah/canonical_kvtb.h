/* Repository-owned reader for the isolated canonical KVTB1 diagnostic artifact. */
#ifndef CANONICAL_KVTB_H
#define CANONICAL_KVTB_H
#include "rules.h"
int canonical_tablebase_load(const char *path);
void canonical_tablebase_close(void);
int canonical_tablebase_known(int active_stones);
int canonical_tablebase_value(const position *p);
long long canonical_tablebase_hits(void);
#endif
