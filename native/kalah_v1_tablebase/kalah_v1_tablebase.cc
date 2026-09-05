// Isolated canonical kalah_v1 feasibility prototype. No production dependencies.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/resource.h>
#include <unistd.h>
#include <string>
#include <vector>

#define main(...) legacy_main(__VA_ARGS__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

using Pits = std::array<uint8_t, 12>;
constexpr int8_t kUnknown = -128;
constexpr uint64_t kMaxTier = 20;
constexpr uint16_t kSchema = 1;
constexpr uint32_t kGeneratorRevision = 2;

// A small self-contained SHA-256 avoids platform and C++ ABI dependencies.
struct Sha256 {
  std::array<uint32_t, 8> state{0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  std::array<uint8_t, 64> block{}; uint64_t bits = 0; size_t used = 0;
  static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
  void transform() { static constexpr uint32_t k[] = {0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U}; uint32_t w[64];
    for (int i=0;i<16;++i) w[i]=(uint32_t(block[4*i])<<24)|(uint32_t(block[4*i+1])<<16)|(uint32_t(block[4*i+2])<<8)|block[4*i+3];
    for (int i=16;i<64;++i) w[i]=(rotr(w[i-15],7)^rotr(w[i-15],18)^(w[i-15]>>3))+w[i-16]+(rotr(w[i-2],17)^rotr(w[i-2],19)^(w[i-2]>>10))+w[i-7];
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3],e=state[4],f=state[5],g=state[6],h=state[7];
    for(int i=0;i<64;++i) { uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g), t1=h+s1+ch+k[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c), t2=s0+maj; h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
  }
  void update(const std::vector<int8_t>& bytes) { for (int8_t v : bytes) { block[used++]=static_cast<uint8_t>(v); bits+=8; if(used==64){transform();used=0;} } }
  std::array<uint8_t,32> finish() { block[used++]=0x80; if(used>56){while(used<64)block[used++]=0;transform();used=0;} while(used<56)block[used++]=0; for(int i=7;i>=0;--i)block[used++]=uint8_t(bits>>(8*i)); transform(); std::array<uint8_t,32> out{}; for(int i=0;i<8;++i)for(int j=0;j<4;++j)out[4*i+j]=uint8_t(state[i]>>(24-8*j)); return out; }
};

uint64_t choose(unsigned n, unsigned k) { if(k>n)return 0;k=std::min(k,n-k);__uint128_t v=1;for(unsigned i=1;i<=k;++i)v=v*(n-k+i)/i;if(v>std::numeric_limits<uint64_t>::max())std::exit(2);return static_cast<uint64_t>(v); }
uint64_t count(unsigned stones) { return choose(stones+11,11); }
uint64_t rank(const Pits& pits) { uint64_t result=0;unsigned remaining=0;for(uint8_t pit:pits)remaining+=pit;for(unsigned i=0;i<11;++i){for(unsigned v=0;v<pits[i];++v)result+=choose(remaining-v+10-i,10-i);remaining-=pits[i];}return result; }
Pits unrank(unsigned stones,uint64_t index) { Pits pits{};unsigned remaining=stones;for(unsigned i=0;i<11;++i)for(unsigned v=0;v<=remaining;++v){uint64_t block=choose(remaining-v+10-i,10-i);if(index<block){pits[i]=v;remaining-=v;break;}index-=block;}pits[11]=remaining;return pits; }
int sum(const Pits& pits,int begin,int end) { int r=0;for(int i=begin;i<end;++i)r+=pits[i];return r; }
struct Transition { Pits pits;int player;int delta;bool extra;int capture;bool terminal;int sweep; };
Transition play(Pits pits,int player,int move) { int absolute=player*6+move,seeds=pits[absolute],index=absolute,owner=player;pits[absolute]=0;bool raw_extra=false;int delta=0;for(int n=0;n<seeds;++n){int next=(index+1)%12,next_owner=next/6;if(owner!=next_owner){bool own_store=owner==player;owner=next_owner;if(own_store){delta+=player==0?1:-1;raw_extra=true;continue;}}raw_extra=false;index=next;++pits[index];}int capture=0;if(!raw_extra){if(index/6==player&&pits[index]==1&&pits[11-index]>0){capture=pits[index]+pits[11-index];pits[index]=pits[11-index]=0;delta+=player==0?capture:-capture;}player=1-player;}bool terminal=sum(pits,player*6,player*6+6)==0;int sweep=0;if(terminal){int opposite=1-player,swept=sum(pits,opposite*6,opposite*6+6);sweep=opposite==0?swept:-swept;delta+=sweep;pits.fill(0);}return {pits,player,delta,raw_extra&&!terminal,capture,terminal,sweep}; }

struct Tables { std::vector<std::vector<int8_t>> values;std::vector<std::vector<uint8_t>> marks;uint64_t edges=0,same_edges=0,lower_edges=0,cycles=0;explicit Tables(unsigned top):values(top+1),marks(top+1){for(unsigned t=0;t<=top;++t){values[t].assign(2*count(t),kUnknown);marks[t].assign(2*count(t),0);}}int8_t solve(const Pits& pits,int player){unsigned t=sum(pits,0,12);uint64_t key=2*rank(pits)+player;if(values[t][key]!=kUnknown)return values[t][key];if(marks[t][key]==1){++cycles;return kUnknown;}marks[t][key]=1;int begin=player*6,best=player==0?-127:127;bool legal=false;for(int move=0;move<6;++move)if(pits[begin+move]){legal=true;++edges;Transition tr=play(pits,player,move);unsigned child_t=sum(tr.pits,0,12);if(child_t==t)++same_edges;else ++lower_edges;int value=tr.delta;if(!tr.terminal){int8_t child=solve(tr.pits,tr.player);if(child==kUnknown)return kUnknown;value+=child;}best=player==0?std::max(best,value):std::min(best,value);}if(!legal)best=sum(pits,0,6)-sum(pits,6,12);marks[t][key]=2;values[t][key]=static_cast<int8_t>(best);return values[t][key];} };

void put16(std::vector<uint8_t>& out,uint16_t v){out.push_back(uint8_t(v));out.push_back(uint8_t(v>>8));} void put32(std::vector<uint8_t>& out,uint32_t v){for(int i=0;i<4;++i)out.push_back(uint8_t(v>>(8*i)));} void put64(std::vector<uint8_t>& out,uint64_t v){for(int i=0;i<8;++i)out.push_back(uint8_t(v>>(8*i)));}
bool get16(const std::vector<uint8_t>& in,size_t& p,uint16_t& v){if(p>in.size()||in.size()-p<2)return false;v=uint16_t(in[p])|(uint16_t(in[p+1])<<8);p+=2;return true;} bool get32(const std::vector<uint8_t>& in,size_t& p,uint32_t& v){if(p>in.size()||in.size()-p<4)return false;v=0;for(int i=0;i<4;++i)v|=uint32_t(in[p++])<<(8*i);return true;} bool get64(const std::vector<uint8_t>& in,size_t& p,uint64_t& v){if(p>in.size()||in.size()-p<8)return false;v=0;for(int i=0;i<8;++i)v|=uint64_t(in[p++])<<(8*i);return true;}
struct File { unsigned tier=0;std::vector<int8_t> payload;std::vector<uint64_t> offsets; };
bool write_file(const std::string& path,unsigned tier,const std::vector<int8_t>& payload){if(tier>kMaxTier)return false;uint64_t states=0;for(unsigned t=0;t<=tier;++t)states+=2*count(t);if(payload.size()!=states)return false;std::vector<uint8_t> h={'K','V','T','B','1'};put16(h,kSchema);h.insert(h.end(),{'k','a','l','a','h','_','v','1'});h.push_back(1);h.push_back(1);h.push_back(static_cast<uint8_t>(kUnknown));h.push_back(static_cast<uint8_t>(tier));put32(h,kGeneratorRevision);h.push_back(static_cast<uint8_t>(kMaxTier));uint64_t header_size=5+2+8+1+1+1+1+4+1+8+8+8+(tier+1)*16+32;put64(h,states);put64(h,states);put64(h,header_size+states);uint64_t offset=0;for(unsigned t=0;t<=tier;++t){put64(h,2*count(t));put64(h,offset);offset+=2*count(t);}Sha256 sha;sha.update(payload);auto digest=sha.finish();h.insert(h.end(),digest.begin(),digest.end());if(h.size()!=header_size)return false;std::ofstream out(path,std::ios::binary);out.write(reinterpret_cast<const char*>(h.data()),static_cast<std::streamsize>(h.size()));out.write(reinterpret_cast<const char*>(payload.data()),static_cast<std::streamsize>(payload.size()));return bool(out);}
bool read_file(const std::string& path,File& file){std::ifstream input(path,std::ios::binary);if(!input)return false;std::vector<uint8_t> data((std::istreambuf_iterator<char>(input)),{});if(input.bad())return false;size_t p=0;if(data.size()<5||std::string(reinterpret_cast<char*>(data.data()),5)!="KVTB1")return false;p=5;uint16_t schema;if(!get16(data,p,schema)||schema!=kSchema||p>data.size()||data.size()-p<8||std::string(reinterpret_cast<char*>(data.data()+p),8)!="kalah_v1")return false;p+=8;if(p>data.size()||data.size()-p<4||data[p++]!=1||data[p++]!=1||static_cast<int8_t>(data[p++])!=kUnknown)return false;File parsed;parsed.tier=data[p++];uint32_t revision;if(!get32(data,p,revision)||revision!=kGeneratorRevision||p>=data.size()||data[p++]!=kMaxTier||parsed.tier>kMaxTier)return false;uint64_t states,payload_bytes,total;if(!get64(data,p,states)||!get64(data,p,payload_bytes)||!get64(data,p,total)||payload_bytes!=states||total!=data.size())return false;parsed.offsets.resize(parsed.tier+1);uint64_t expected=0;for(unsigned t=0;t<=parsed.tier;++t){uint64_t tier_states,offset;if(!get64(data,p,tier_states)||!get64(data,p,offset)||tier_states!=2*count(t)||offset!=expected||expected>std::numeric_limits<uint64_t>::max()-tier_states)return false;parsed.offsets[t]=offset;expected+=tier_states;}if(expected!=states||payload_bytes>std::numeric_limits<size_t>::max()||p>data.size()||data.size()-p<32)return false;std::array<uint8_t,32> digest{};std::copy_n(data.begin()+static_cast<long>(p),32,digest.begin());p+=32;if(payload_bytes!=data.size()-p)return false;parsed.payload.resize(static_cast<size_t>(payload_bytes));for(size_t i=0;i<parsed.payload.size();++i)parsed.payload[i]=static_cast<int8_t>(data[p+i]);Sha256 sha;sha.update(parsed.payload);if(sha.finish()!=digest)return false;file=std::move(parsed);return true;}

std::vector<int> numbers(const std::string& line,const std::string& name){size_t p=line.find("\""+name+"\"");if(p==std::string::npos)return{};p=line.find('[',p);size_t q=line.find(']',p);std::vector<int> out;int n=0;bool in=false;for(size_t i=p+1;i<q;++i){if(line[i]>='0'&&line[i]<='9'){n=n*10+line[i]-'0';in=true;}else if(in){out.push_back(n);n=0;in=false;}}if(in)out.push_back(n);return out;}int number(const std::string& line,const std::string& name){auto p=line.find("\""+name+"\"");if(p==std::string::npos)return-1;p=line.find(':',p);return std::atoi(line.c_str()+p+1);}
void emit_transition(const Pits& pits,int player,int move){Transition t=play(pits,player,move);std::cout<<"{\"pits\":[";for(int i=0;i<12;++i)std::cout<<(i?",":"")<<int(t.pits[i]);std::cout<<"],\"player\":"<<t.player<<",\"delta\":"<<t.delta<<",\"extra\":"<<(t.extra?"true":"false")<<",\"capture\":"<<t.capture<<",\"terminal\":"<<(t.terminal?"true":"false")<<",\"sweep\":"<<t.sweep<<"}\n";}
int benchmark(unsigned top){auto start=std::chrono::steady_clock::now();Tables tables(top);for(unsigned t=0;t<=top;++t)for(uint64_t r=0;r<count(t);++r)for(int p=0;p<2;++p)if(tables.solve(unrank(t,r),p)==kUnknown){std::cerr<<"cycle\n";return 3;}double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();std::cout<<"{\"max_tier\":"<<top<<",\"states\":";uint64_t states=0;for(unsigned t=0;t<=top;++t)states+=2*count(t);std::cout<<states<<",\"edges\":"<<tables.edges<<",\"seconds\":"<<seconds<<"}\n";return 0;}
int main(int argc,char** argv){if(argc==2&&std::string(argv[1])=="format-selftest"){std::vector<int8_t> abc={'a','b','c'};Sha256 sha;sha.update(abc);const std::array<uint8_t,32> expected_sha={0xba,0x78,0x16,0xbf,0x8f,0x01,0xcf,0xea,0x41,0x41,0x40,0xde,0x5d,0xae,0x22,0x23,0xb0,0x03,0x61,0xa3,0x96,0x17,0x7a,0x9c,0xb4,0x10,0xff,0x61,0xf2,0x00,0x15,0xad};if(sha.finish()!=expected_sha)return 6;std::vector<int8_t> p={1,-2};std::array<char,64> path{};std::snprintf(path.data(),path.size(),"%s","/tmp/kalah_v1_format_selftest.XXXXXX");int fd=mkstemp(path.data());if(fd<0)return 6;close(fd);std::string f=path.data();File parsed;bool ok=write_file(f,0,p)&&read_file(f,parsed)&&parsed.payload==p&&!write_file(f,0,{1,-2,3});if(ok){std::ofstream out(f,std::ios::app|std::ios::binary);out.put(0);out.close();ok=!read_file(f,parsed);}std::remove(f.c_str());return ok?0:6;}if(argc==3&&std::string(argv[1])=="benchmark"){unsigned top=std::strtoul(argv[2],nullptr,10);return top>kMaxTier?2:benchmark(top);}if(argc==4&&std::string(argv[1])=="generate"){unsigned top=std::strtoul(argv[2],nullptr,10);if(top>kMaxTier)return 2;Tables tables(top);for(unsigned t=0;t<=top;++t)for(uint64_t r=0;r<count(t);++r)for(int p=0;p<2;++p)if(tables.solve(unrank(t,r),p)==kUnknown){std::cerr<<"cycle\n";return 3;}std::vector<int8_t> payload;for(unsigned t=0;t<=top;++t)payload.insert(payload.end(),tables.values[t].begin(),tables.values[t].end());if(!write_file(argv[3],top,payload))return 4;Sha256 sha;sha.update(payload);auto d=sha.finish();std::cout<<"{\"classification\":\"ok\",\"max_tier\":"<<top<<",\"states\":"<<payload.size()<<",\"edges\":"<<tables.edges<<",\"same_tier_edges\":"<<tables.same_edges<<",\"lower_tier_edges\":"<<tables.lower_edges<<",\"cycles\":"<<tables.cycles<<",\"payload_sha256\":\"";for(uint8_t b:d)std::cout<<"0123456789abcdef"[b>>4]<<"0123456789abcdef"[b&15];std::cout<<"\"}\n";return 0;}if(argc==3&&std::string(argv[1])=="probe"){File file;if(!read_file(argv[2],file))return 5;std::string line;while(std::getline(std::cin,line)){auto pv=numbers(line,"pits");int player=number(line,"player");if(pv.size()!=12||player<0||player>1){std::cout<<"{\"error\":\"invalid request\"}\n";continue;}Pits pits{};int stones=0;for(int i=0;i<12;++i){if(pv[i]>255){std::cout<<"{\"error\":\"pit overflow\"}\n";goto next_probe;}pits[i]=pv[i];stones+=pv[i];}if(stones>int(file.tier)){std::cout<<"{\"error\":\"tier unavailable\"}\n";goto next_probe;}std::cout<<"{\"value\":"<<int(file.payload[file.offsets[stones]+2*rank(pits)+player])<<",\"actions\":{";{bool first=true;for(int m=0;m<6;++m)if(pits[player*6+m]){Transition tr=play(pits,player,m);int v=tr.delta+(tr.terminal?0:file.payload[file.offsets[sum(tr.pits,0,12)]+2*rank(tr.pits)+tr.player]);std::cout<<(first?"":",")<<"\""<<m<<"\":"<<v;first=false;}}std::cout<<"}}\n";next_probe:;}return 0;}std::string line;while(std::getline(std::cin,line)){auto pv=numbers(line,"pits");int player=number(line,"player"),move=number(line,"move");if(pv.size()!=12||player<0||player>1||move<0||move>5){std::cout<<"{\"error\":\"invalid request\"}\n";continue;}Pits pits{};for(int i=0;i<12;++i){if(pv[i]>255){std::cout<<"{\"error\":\"pit overflow\"}\n";goto next_transition;}pits[i]=pv[i];}if(!pits[player*6+move])std::cout<<"{\"error\":\"illegal move\"}\n";else emit_transition(pits,player,move);next_transition:;} }
uint64_t next_random(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

struct LookupQuery {
  Pits pits;
  int player;
};

int lookup_benchmark(const std::string& path, uint64_t query_count, uint64_t seed) {
  auto load_start = std::chrono::steady_clock::now();
  File file;
  if (!read_file(path, file)) return 5;
  uint64_t load_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - load_start).count();

  if (query_count > std::numeric_limits<size_t>::max()) return 2;
  std::vector<LookupQuery> queries;
  queries.reserve(static_cast<size_t>(query_count));
  uint64_t random_state = seed;
  for (uint64_t i = 0; i < query_count; ++i) {
    unsigned tier = static_cast<unsigned>(next_random(random_state) % (file.tier + 1));
    uint64_t index = next_random(random_state) % count(tier);
    queries.push_back({unrank(tier, index), static_cast<int>(next_random(random_state) & 1)});
  }
  // The corpus hash covers the exact binary records looked up below: 12 pit
  // counts followed by the current-player byte for each deterministic query.
  Sha256 corpus_sha;
  for (const LookupQuery& query : queries) {
    std::vector<int8_t> record;
    record.reserve(13);
    for (uint8_t pit : query.pits) record.push_back(static_cast<int8_t>(pit));
    record.push_back(static_cast<int8_t>(query.player));
    corpus_sha.update(record);
  }
  auto corpus_digest = corpus_sha.finish();

  volatile int64_t checksum = 0;
  auto cold_start = std::chrono::steady_clock::now();
  for (const LookupQuery& query : queries) {
    unsigned tier = static_cast<unsigned>(sum(query.pits, 0, 12));
    checksum += file.payload[file.offsets[tier] + 2 * rank(query.pits) + query.player];
  }
  uint64_t cold_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - cold_start).count();

  std::vector<uint64_t> warm_latencies;
  warm_latencies.reserve(queries.size());
  auto warm_start = std::chrono::steady_clock::now();
  for (const LookupQuery& query : queries) {
    auto start = std::chrono::steady_clock::now();
    unsigned tier = static_cast<unsigned>(sum(query.pits, 0, 12));
    checksum += file.payload[file.offsets[tier] + 2 * rank(query.pits) + query.player];
    warm_latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
  }
  uint64_t warm_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - warm_start).count();
  std::sort(warm_latencies.begin(), warm_latencies.end());
  uint64_t median_ns = warm_latencies[(warm_latencies.size() - 1) / 2];
  uint64_t p95_ns = warm_latencies[(warm_latencies.size() - 1) * 95 / 100];
  struct rusage usage {};
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 7;
  std::cout << "{\"query_count\":" << query_count
            << ",\"seed\":" << seed << ",\"load_ns\":" << load_ns
            << ",\"cold_lookup_ns\":" << cold_ns
            << ",\"warm_lookup_count\":" << warm_latencies.size()
            << ",\"warm_lookup_ns\":" << warm_ns
            << ",\"warm_median_lookup_ns\":" << median_ns
            << ",\"warm_p95_lookup_ns\":" << p95_ns
            << ",\"rss_kib\":" << usage.ru_maxrss
            << ",\"corpus_sha256\":\"";
  for (uint8_t byte : corpus_digest) std::cout << "0123456789abcdef"[byte >> 4] << "0123456789abcdef"[byte & 15];
  std::cout << "\""
            << ",\"checksum\":" << checksum << "}\n";
  return 0;
}

bool parse_uint64(const char* text, uint64_t& value) {
  if (!*text || std::isspace(static_cast<unsigned char>(*text)) || *text == '-') return false;
  char* end = nullptr;
  errno = 0;
  unsigned long long parsed = std::strtoull(text, &end, 10);
  if (errno == ERANGE || *end) return false;
  value = parsed;
  return true;
}

#pragma GCC diagnostic pop

int transition_main() {
  std::string line;
  while (std::getline(std::cin, line)) {
    auto pv = numbers(line, "pits");
    int player = number(line, "player"), move = number(line, "move");
    if (pv.size() != 12 || player < 0 || player > 1 || move < 0 || move > 5) {
      std::cout << "{\"error\":\"invalid request\"}\n";
      continue;
    }
    Pits pits{};
    for (int i = 0; i < 12; ++i) {
      if (pv[i] > 255) {
        std::cout << "{\"error\":\"pit overflow\"}\n";
        goto next_transition;
      }
      pits[i] = pv[i];
    }
    if (!pits[player * 6 + move]) std::cout << "{\"error\":\"illegal move\"}\n";
    else emit_transition(pits, player, move);
  next_transition:;
  }
  return 0;
}

int (main)(int argc, char** argv) {
  if (argc == 5 && std::string(argv[1]) == "lookup-benchmark") {
    uint64_t query_count, seed;
    if (!parse_uint64(argv[3], query_count) || !parse_uint64(argv[4], seed) || !query_count) return 2;
    return lookup_benchmark(argv[2], query_count, seed);
  }
  if (argc == 1) return transition_main();
  return legacy_main(argc, argv);
}
