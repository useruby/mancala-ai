#include "canonical_kvtb.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *payload;
static uint64_t offsets[21];
static int top = -1;
static long long hits;

static uint64_t get64(const unsigned char *p) { uint64_t v=0; int i; for(i=0;i<8;i++) v|=(uint64_t)p[i]<<(8*i); return v; }
static uint64_t choose(unsigned n,unsigned k) { uint64_t v=1; unsigned i; if(k>n)return 0; if(k>n-k)k=n-k; for(i=1;i<=k;i++)v=v*(n-k+i)/i; return v; }
static uint64_t rank_pits(const position *p) { uint64_t r=0; unsigned remaining=0,i,v,x; for(i=0;i<6;i++)remaining+=(unsigned char)p->a[i]+(unsigned char)p->a[i+7]; for(i=0;i<11;i++){x=i<6?(unsigned char)p->a[i]:(unsigned char)p->a[i+1]; for(v=0;v<x;v++)r+=choose(remaining-v+10-i,10-i);remaining-=x;} return r; }
int canonical_tablebase_load(const char *path) { FILE *f; unsigned char h[416]; uint64_t states,bytes; int i; if(!path)return 0; f=fopen(path,"rb"); if(!f)return 0; if(fread(h,1,24,f)!=24 || memcmp(h,"KVTB1\1\0kalah_v1\1\1\200",18)){fclose(f);return 0;} top=h[18]; if(top!=18 || fread(h+24,1,56+16*(top+1)-24,f)!=56+16*(top+1)-24){fclose(f);return 0;} states=get64(h+24);bytes=get64(h+32); if(states!=bytes){fclose(f);return 0;} for(i=0;i<=top;i++)offsets[i]=get64(h+56+16*i+8); if(fseek(f,80+16*(top+1),SEEK_SET)){fclose(f);return 0;} payload=malloc((size_t)bytes); if(!payload || fread(payload,1,(size_t)bytes,f)!=bytes){free(payload);payload=0;fclose(f);return 0;} fclose(f); hits=0; return 1; }
void canonical_tablebase_close(void) { free(payload); payload=0; top=-1; }
int canonical_tablebase_known(int active_stones) { return payload && active_stones>=0 && active_stones<=top; }
int canonical_tablebase_value(const position *p) { int active=0,i,v; for(i=0;i<6;i++)active+=(unsigned char)p->a[i]+(unsigned char)p->a[i+7]; v=(int)(int8_t)payload[offsets[active]+2*rank_pits(p)+p->s]; hits++; return p->s ? -((int)p->a[6]-(int)p->a[13]+v) : (int)p->a[6]-(int)p->a[13]+v; }
long long canonical_tablebase_hits(void) { return hits; }
