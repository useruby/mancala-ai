// Isolated canonical kalah_v1 feasibility prototype. No production dependencies.
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using Pits = std::array<uint8_t, 12>;
constexpr int8_t kUnknown = -128;
constexpr uint64_t kMaxTier = 20;
struct Header { char magic[5] = {'K','V','T','B','1'}; uint8_t schema = 1; char rules[8] = {'k','a','l','a','h','_','v','1'}; uint8_t tier = 0; uint8_t value_bytes = 1; int8_t unknown = kUnknown; uint8_t endian = 1; uint64_t states = 0; uint64_t checksum = 0; };
uint64_t fnv1a(const std::vector<int8_t>& bytes) { uint64_t h=1469598103934665603ULL; for(int8_t b:bytes) { h^=static_cast<uint8_t>(b); h*=1099511628211ULL; } return h; }

uint64_t choose(unsigned n, unsigned k) {
  if (k > n) return 0;
  k = std::min(k, n - k);
  __uint128_t value = 1;
  for (unsigned i = 1; i <= k; ++i) value = value * (n - k + i) / i;
  if (value > std::numeric_limits<uint64_t>::max()) std::exit(2);
  return static_cast<uint64_t>(value);
}
uint64_t count(unsigned stones) { return choose(stones + 11, 11); }
uint64_t rank(const Pits& pits) {
  uint64_t result = 0; unsigned remaining = 0;
  for (uint8_t pit : pits) remaining += pit;
  for (unsigned i = 0; i < 11; ++i) {
    for (unsigned v = 0; v < pits[i]; ++v) result += choose(remaining - v + 10 - i, 10 - i);
    remaining -= pits[i];
  }
  return result;
}
Pits unrank(unsigned stones, uint64_t index) {
  Pits pits{}; unsigned remaining = stones;
  for (unsigned i = 0; i < 11; ++i) {
    for (unsigned v = 0; v <= remaining; ++v) {
      uint64_t block = choose(remaining - v + 10 - i, 10 - i);
      if (index < block) { pits[i] = v; remaining -= v; break; }
      index -= block;
    }
  }
  pits[11] = remaining;
  return pits;
}
int sum(const Pits& pits, int begin, int end) { int r = 0; for (int i=begin;i<end;++i) r += pits[i]; return r; }
struct Transition { Pits pits; int player; int delta; bool extra; int capture; bool terminal; int sweep; };
Transition play(Pits pits, int player, int move) {
  int absolute = player * 6 + move, seeds = pits[absolute], index = absolute, owner = player;
  pits[absolute] = 0; bool raw_extra = false; int delta = 0;
  for (int n = 0; n < seeds; ++n) {
    int next = (index + 1) % 12, next_owner = next / 6;
    if (owner != next_owner) { bool own_store = owner == player; owner = next_owner; if (own_store) { delta += player == 0 ? 1 : -1; raw_extra = true; continue; } }
    raw_extra = false; index = next; ++pits[index];
  }
  int capture = 0;
  if (!raw_extra) {
    if (index / 6 == player && pits[index] == 1 && pits[11-index] > 0) {
      capture = pits[index] + pits[11-index]; pits[index] = pits[11-index] = 0;
      delta += player == 0 ? capture : -capture;
    }
    player = 1 - player;
  }
  bool terminal = sum(pits, player*6, player*6+6) == 0;
  int sweep = 0;
  if (terminal) {
    int opposite = 1-player, swept = sum(pits, opposite*6, opposite*6+6);
    sweep = opposite == 0 ? swept : -swept;
    delta += sweep; pits.fill(0);
  }
  return {pits, player, delta, raw_extra && !terminal, capture, terminal, sweep};
}

struct Tables {
  std::vector<std::vector<int8_t>> values;
  std::vector<std::vector<uint8_t>> marks;
  uint64_t edges = 0, same_edges = 0, lower_edges = 0, cycles = 0;
  explicit Tables(unsigned top) : values(top+1), marks(top+1) {
    for (unsigned t=0;t<=top;++t) { values[t].assign(2*count(t), kUnknown); marks[t].assign(2*count(t), 0); }
  }
  int8_t solve(const Pits& pits, int player) {
    unsigned t = sum(pits,0,12); uint64_t key = 2*rank(pits)+player;
    if (values[t][key] != kUnknown) return values[t][key];
    if (marks[t][key] == 1) { ++cycles; return kUnknown; }
    marks[t][key] = 1;
    int begin=player*6, best=player==0 ? -127 : 127; bool legal=false;
    for(int move=0;move<6;++move) if(pits[begin+move]) {
      legal=true; ++edges; Transition tr=play(pits,player,move); unsigned child_t=sum(tr.pits,0,12);
      if(child_t==t) ++same_edges; else ++lower_edges;
      int value=tr.delta;
      if(!tr.terminal) { int8_t child=solve(tr.pits,tr.player); if(child==kUnknown) return kUnknown; value+=child; }
      best=player==0?std::max(best,value):std::min(best,value);
    }
    if(!legal) best=sum(pits,0,6)-sum(pits,6,12);
    marks[t][key] = 2; values[t][key]=static_cast<int8_t>(best); return values[t][key];
  }
};

std::vector<int> numbers(const std::string& line, const std::string& name) {
  size_t p=line.find("\""+name+"\""); if(p==std::string::npos) return {};
  p=line.find('[',p); size_t q=line.find(']',p); std::vector<int> out; int n=0; bool in=false;
  for(size_t i=p+1;i<q;++i) { if(std::isdigit(line[i])) { n=n*10+line[i]-'0'; in=true; } else if(in) {out.push_back(n);n=0;in=false;} } if(in) out.push_back(n); return out;
}
int number(const std::string& line, const std::string& name) { auto p=line.find("\""+name+"\""); if(p==std::string::npos) return -1; p=line.find(':',p); return std::atoi(line.c_str()+p+1); }
void emit_transition(const Pits& pits,int player,int move) {
  Transition t=play(pits,player,move); std::cout<<"{\"pits\":[";
  for(int i=0;i<12;++i) std::cout<<(i?",":"")<<int(t.pits[i]);
  std::cout<<"],\"player\":"<<t.player<<",\"delta\":"<<t.delta<<",\"extra\":"<<(t.extra?"true":"false")<<",\"capture\":"<<t.capture<<",\"terminal\":"<<(t.terminal?"true":"false")<<",\"sweep\":"<<t.sweep<<"}\n";
}
int main(int argc,char** argv) {
  if(argc==4 && std::string(argv[1])=="generate") {
    unsigned top=std::strtoul(argv[2],nullptr,10); if(top>kMaxTier) return 2; Tables tables(top);
    for(unsigned t=0;t<=top;++t) for(uint64_t r=0;r<count(t);++r) for(int p=0;p<2;++p) if(tables.solve(unrank(t,r),p)==kUnknown) { std::cerr<<"cycle\n"; return 3; }
    std::vector<int8_t> payload; uint64_t cumulative=0; for(unsigned t=0;t<=top;++t) { cumulative+=2*count(t); payload.insert(payload.end(),tables.values[t].begin(),tables.values[t].end()); }
    Header header; header.tier=top; header.states=cumulative; header.checksum=fnv1a(payload);
    std::ofstream output(argv[3],std::ios::binary); if(!output) return 4;
    output.write(reinterpret_cast<const char*>(&header),sizeof(header)); output.write(reinterpret_cast<const char*>(payload.data()),payload.size());
    if(!output) return 4;
    std::cout<<"{\"classification\":\"ok\",\"max_tier\":"<<top<<",\"states\":"<<cumulative<<",\"edges\":"<<tables.edges<<",\"same_tier_edges\":"<<tables.same_edges<<",\"lower_tier_edges\":"<<tables.lower_edges<<",\"cycles\":"<<tables.cycles<<",\"checksum\":\""<<header.checksum<<"\"}\n"; return 0;
  }
  if(argc==3 && std::string(argv[1])=="probe") {
    std::ifstream input(argv[2],std::ios::binary); Header header; input.read(reinterpret_cast<char*>(&header),sizeof(header));
    if(!input || std::string(header.magic,5)!="KVTB1" || header.schema!=1 || std::string(header.rules,8)!="kalah_v1") return 5;
    std::vector<int8_t> payload(header.states); input.read(reinterpret_cast<char*>(payload.data()),payload.size()); if(!input || fnv1a(payload)!=header.checksum) return 5;
    std::vector<uint64_t> offsets(header.tier+1); uint64_t offset=0; for(unsigned t=0;t<=header.tier;++t) { offsets[t]=offset; offset+=2*count(t); }
    std::string line; while(std::getline(std::cin,line)) { auto pitsv=numbers(line,"pits"); int player=number(line,"player"); if(pitsv.size()!=12||player<0||player>1){std::cout<<"{\"error\":\"invalid request\"}\n";continue;} Pits pits{};int stones=0;for(int i=0;i<12;++i){pits[i]=pitsv[i];stones+=pitsv[i];} if(stones>header.tier){std::cout<<"{\"error\":\"tier unavailable\"}\n";continue;} int value=payload[offsets[stones]+2*rank(pits)+player]; std::cout<<"{\"value\":"<<value<<",\"actions\":{";bool first=true;for(int m=0;m<6;++m)if(pits[player*6+m]){Transition tr=play(pits,player,m);int v=tr.delta+(tr.terminal?0:payload[offsets[sum(tr.pits,0,12)]+2*rank(tr.pits)+tr.player]);std::cout<<(first?"":",")<<"\""<<m<<"\":"<<v;first=false;}std::cout<<"}}\n"; }
    return 0;
  }
  std::string line; while(std::getline(std::cin,line)) {
    auto pitsv=numbers(line,"pits"); int player=number(line,"player"), move=number(line,"move");
    if(pitsv.size()!=12 || player<0 || player>1 || move<0 || move>5) { std::cout<<"{\"error\":\"invalid request\"}\n"; continue; }
    Pits pits{}; for(int i=0;i<12;++i) {if(pitsv[i]>255){std::cout<<"{\"error\":\"pit overflow\"}\n";goto next;} pits[i]=pitsv[i];}
    if(!pits[player*6+move]) std::cout<<"{\"error\":\"illegal move\"}\n"; else emit_transition(pits,player,move);
    next: ;
  }
}
