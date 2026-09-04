#include <assert.h>
#include <string.h>
#define __hash_lowlevel
#include "mix.h"
static ub8 key(position p) { long a=packpits(p.a),b=packpits(p.a+7);ub8 k;set_ub8(k,a,b);return k; }
int main(void) { char guard[]={99,0,0,0,0,0,0,77}, a[]={1,2,3,4,5,6}, b[]={1,2,3,4,5,6};position p,q;int i;assert(packpits(guard+1)==0);assert(packpits(a)==1L|(2L<<5)|(3L<<10)|(4L<<15)|(5L<<20)|(6L<<25));for(i=0;i<6;i++){long k=packpits(b);b[i]++;assert(packpits(b)!=k);b[i]--;}a[4]=32;assert(packpits(a)==HASHOVERFLOW);memset(&p,0,sizeof(p));memset(&q,0,sizeof(q));p.a[0]=1;q.a[1]=1;assert(key(p)!=key(q));return 0;}
