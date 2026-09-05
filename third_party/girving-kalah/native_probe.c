/* Repository-owned JSON Lines adapter for the pinned Girving Kalah source. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "canonical_kvtb.h"
#include "crunch.cilkh"

static const char *at(const char *s, const char *key) { const char *p = strstr(s, key); return p ? strchr(p, ':') : 0; }
static int integer(const char *s, const char *key, int *v) { char *e; const char *p = at(s,key); if (!p) return 0; *v = strtol(p+1,&e,10); return e != p+1; }
static int array(const char *s, const char *key, int *v, int n) { const char *p=at(s,key); int i; char *e; if (!p || !(p=strchr(p,'['))) return 0; for(i=0,p++;i<n;i++,p=e+1) { v[i]=strtol(p,&e,10); if(e==p || (i+1<n && *e!=',')) return 0; } return 1; }
static void readp(const char *s, position *p) { int a[12],b[2],i,j; if(!array(s,"\"pits\"",a,12)||!array(s,"\"stores\"",b,2)||!integer(s,"\"player\"",&i)||i<0||i>1) exit(2); for(j=0;j<6;j++){p->a[j]=a[j];p->a[j+7]=a[j+6];} p->a[6]=b[0]; p->a[13]=b[1]; p->s=i;p->w=-1; }
static int value(position p) { char m[MAXMVC]; int d,r; if(p.w>=0)return p.a[6]-p.a[13]; init_stat(); r=solve(m,p,200,&d,0,SF_GUESS|SF_STALE); return p.s ? -r:r; }
static void board(position p, int x) { int i,margin=p.a[6]-p.a[13]; for(i=0;i<6;i++)margin+=p.a[i]-p.a[i+7]; printf("\"pits\":["); for(i=0;i<6;i++)printf("%s%d",i?",":"",p.a[i]);for(i=0;i<6;i++)printf(",%d",p.a[i+7]);printf("],\"stores\":[%d,%d],\"player\":%d,\"extra_turn\":%s,\"terminal\":%s,\"final_margin\":%d",p.a[6],p.a[13],p.s,x && p.w < 0 ? "true":"false",p.w>=0?"true":"false",margin); }
static void apply(const char *s) { position p;int a,x,extra;readp(s,&p);if(!integer(s,"\"action\"",&a)||a<0||a>=6||!bin(p,a))exit(2);extra=(a+bin(p,a))%13==6;x=move(&p,a);if(p.w>=0&&!extra)p.s=1-p.s;printf("{\"operation\":\"apply\",");board(p,extra);puts("}"); }
static void solveone(const char *s) { position p;readp(s,&p);printf("{\"operation\":\"solve\",");board(p,0);printf(",\"exact_value\":%d}\n",value(p)); }
static void label(const char *s) { position p,c;int a,x,v,best,first=1;readp(s,&p);best=p.s?10000:-10000;printf("{\"operation\":\"label\",");board(p,0);printf(",\"action_values\":{");for(a=0;a<6;a++)if(bin(p,a)){c=p;x=move(&c,a);v=c.w>=0?c.a[6]-c.a[13]:value(c);if((!p.s&&v>best)||(p.s&&v<best))best=v;printf("%s\"%d\":%d",first?"":",",a,v);first=0;}printf("},\"exact_value\":%d,\"optimal_actions\":[",best);first=1;for(a=0;a<6;a++)if(bin(p,a)){c=p;x=move(&c,a);v=c.w>=0?c.a[6]-c.a[13]:value(c);if(v==best){printf("%s%d",first?"":",",a);first=0;}}printf("],\"action_margins\":{");first=1;for(a=0;a<6;a++)if(bin(p,a)){c=p;x=move(&c,a);v=c.w>=0?c.a[6]-c.a[13]:value(c);printf("%s\"%d\":%d",first?"":",",a,p.s ? v-best : best-v);first=0;}puts("}}"); }
int main(void) { char s[4096],*tablebase=getenv("NATIVE_CANONICAL_KVTB");init_hash(16,0);if(tablebase&&!canonical_tablebase_load(tablebase))return 3;while(fgets(s,sizeof(s),stdin)){if(strstr(s,"apply"))apply(s);else if(strstr(s,"solve"))solveone(s);else if(strstr(s,"label"))label(s);else return 2;fflush(stdout);}canonical_tablebase_close();close_hash();return 0; }
