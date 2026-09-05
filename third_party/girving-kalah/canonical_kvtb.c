#include "canonical_kvtb.h"
#include <limits.h>
#include <openssl/sha.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KVTB_SCHEMA 1
#define KVTB_MAX_TIER 20
#define KVTB_REVISION 2
#define KVTB_UNKNOWN ((int8_t)-128)

static unsigned char *payload;
static uint64_t offsets[KVTB_MAX_TIER + 1];
static int top = -1;
static long long hits;
static long long request_lookups, request_hits, tier_hits[KVTB_MAX_TIER + 1];
static int first_hit_depth, min_hit_tier, max_hit_tier;

static uint64_t get64(const unsigned char *p) { uint64_t v=0; int i; for(i=0;i<8;i++) v|=(uint64_t)p[i]<<(8*i); return v; }
static uint32_t get32(const unsigned char *p) { uint32_t v=0; int i; for(i=0;i<4;i++) v|=(uint32_t)p[i]<<(8*i); return v; }
static uint64_t choose(unsigned n,unsigned k) { uint64_t v=1; unsigned i; if(k>n)return 0; if(k>n-k)k=n-k; for(i=1;i<=k;i++)v=v*(n-k+i)/i; return v; }
static uint64_t count(unsigned tier) { return choose(tier+11,11); }
static uint64_t rank_pits(const position *p) { uint64_t r=0; unsigned remaining=0,i,v,x; for(i=0;i<6;i++)remaining+=(unsigned char)p->a[i]+(unsigned char)p->a[i+7]; for(i=0;i<11;i++){x=i<6?(unsigned char)p->a[i]:(unsigned char)p->a[i+1]; for(v=0;v<x;v++)r+=choose(remaining-v+10-i,10-i);remaining-=x;} return r; }

void canonical_tablebase_reset_request_metrics(void) { request_lookups=request_hits=0; memset(tier_hits,0,sizeof(tier_hits)); first_hit_depth=-1; min_hit_tier=KVTB_MAX_TIER+1; max_hit_tier=-1; }
void canonical_tablebase_close(void) { free(payload); payload=0; memset(offsets,0,sizeof(offsets)); top=-1; hits=0; canonical_tablebase_reset_request_metrics(); }

int canonical_tablebase_load(const char *path) {
  FILE *f=0; unsigned char h[416],digest[32]; uint64_t states,bytes,total,expected=0,header; long length; int i;
  canonical_tablebase_close();
  if(!path || !(f=fopen(path,"rb"))) goto fail;
  if(fseek(f,0,SEEK_END) || (length=ftell(f))<0 || fseek(f,0,SEEK_SET)) goto fail;
  if((uintmax_t)length>UINT64_MAX || fread(h,1,48,f)!=48) goto fail;
  if(memcmp(h,"KVTB1",5) || h[5]!=KVTB_SCHEMA || h[6] || memcmp(h+7,"kalah_v1",8) || h[15]!=1 || h[16]!=1 || (int8_t)h[17]!=KVTB_UNKNOWN) goto fail;
  top=h[18];
  if(top>KVTB_MAX_TIER || get32(h+19)!=KVTB_REVISION || h[23]!=KVTB_MAX_TIER) goto fail;
  states=get64(h+24); bytes=get64(h+32); total=get64(h+40); header=80+16*((uint64_t)top+1);
  if(bytes!=states || total!=(uint64_t)length || header>sizeof(h) || total<header || bytes!=total-header) goto fail;
  if(fread(h+48,1,(size_t)(header-48),f)!=(size_t)(header-48)) goto fail;
  for(i=0;i<=top;i++) { uint64_t tier_states=get64(h+48+16*i),offset=get64(h+56+16*i); uint64_t wanted=2*count((unsigned)i); if(tier_states!=wanted || offset!=expected || expected>UINT64_MAX-tier_states) goto fail; offsets[i]=offset; expected+=tier_states; }
  if(expected!=states || bytes>SIZE_MAX) goto fail;
  memcpy(digest,h+header-32,32);
  if(!(payload=malloc((size_t)bytes))) goto fail;
  if(fread(payload,1,(size_t)bytes,f)!=(size_t)bytes || fgetc(f)!=EOF) goto fail;
  { unsigned char actual[SHA256_DIGEST_LENGTH]; size_t n; if(!SHA256(payload,(size_t)bytes,actual) || memcmp(actual,digest,32)){fprintf(stderr,"canonical tablebase load failed: payload checksum\n");goto fail;} for(n=0;n<(size_t)bytes;n++)if((int8_t)payload[n]==KVTB_UNKNOWN){fprintf(stderr,"canonical tablebase load failed: unknown payload entry\n");goto fail;} }
  fclose(f); hits=0; canonical_tablebase_reset_request_metrics(); return 1;
fail:
  fprintf(stderr,"canonical tablebase load failed: invalid KVTB1 file\n");
  if(f) fclose(f);
  canonical_tablebase_close();
  return 0;
}

int canonical_tablebase_known(int active_stones) { return payload && active_stones>=0 && active_stones<=top; }
void canonical_tablebase_record_lookup(int active_stones, int depth) { request_lookups++; if(!canonical_tablebase_known(active_stones))return; request_hits++; tier_hits[active_stones]++; if(first_hit_depth<0)first_hit_depth=depth; if(active_stones<min_hit_tier)min_hit_tier=active_stones; if(active_stones>max_hit_tier)max_hit_tier=active_stones; }
long long canonical_tablebase_request_lookups(void) { return request_lookups; }
long long canonical_tablebase_request_hits(void) { return request_hits; }
int canonical_tablebase_first_hit_depth(void) { return first_hit_depth; }
int canonical_tablebase_min_hit_tier(void) { return min_hit_tier>KVTB_MAX_TIER ? -1:min_hit_tier; }
int canonical_tablebase_max_hit_tier(void) { return max_hit_tier; }
long long canonical_tablebase_tier_hits(int tier) { return tier>=0 && tier<=KVTB_MAX_TIER?tier_hits[tier]:0; }
int canonical_tablebase_diagnostics(const position *p, canonical_tablebase_info *info) {
  int active=0,i; uint64_t rank,index;
  if(!p || !info) return 0;
  for(i=0;i<6;i++)active+=(unsigned char)p->a[i]+(unsigned char)p->a[i+7];
  if(!canonical_tablebase_known(active) || (p->s!=0 && p->s!=1)) return 0;
  rank=rank_pits(p); if(rank>=count((unsigned)active) || rank>(UINT64_MAX-(uint64_t)p->s)/2) return 0; index=offsets[active]+2*rank+(uint64_t)p->s;
  if(index>=offsets[active]+2*count((unsigned)active) || index>=UINT64_MAX || (int8_t)payload[index]==KVTB_UNKNOWN) return 0;
  info->active_stones=active; info->rank=rank; info->offset=offsets[active]; info->raw_value=(int8_t)payload[index]; info->store_margin=(int)p->a[6]-(int)p->a[13]; info->upstream_value=p->s ? -(info->store_margin+info->raw_value) : info->store_margin+info->raw_value; info->player_zero_value=p->s ? -info->upstream_value : info->upstream_value;
  return 1;
}
int canonical_tablebase_value(const position *p) { canonical_tablebase_info info; if(!canonical_tablebase_diagnostics(p,&info)) return 0; canonical_tablebase_record_lookup(info.active_stones,-1); hits++; return info.upstream_value; }
long long canonical_tablebase_hits(void) { return hits; }
