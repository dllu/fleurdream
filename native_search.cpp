// Bounded discovery only. Python validates returned casts and retains the
// complete branch-and-bound frontier and its optimality proof independently.
// Spell catalog, flower access, alphabet, and ALTER mappings arrive on stdin.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using namespace std;
using U = uint64_t;
enum Op {
  ALTER,
  MOVE,
  ADD,
  ADJ,
  COMBINE,
  CHANGE,
  CHOP,
  DROP,
  SWAP,
  SWAPADJ,
  SUB,
  DUP,
  CHECK,
  RECAST,
  REPLACE,
  MOST,
  UNLOCK
};
const string names[] = {"alter",         "move",         "add",
                        "adjacent",      "combine",      "change",
                        "chop",          "drop",         "swap",
                        "swap_adjacent", "subtract",     "duplicate",
                        "spellcheck",    "recast_spell", "find_and_replace",
                        "most_frequent", "unlock_letter"};
array<string, 26> alterations;
bool structural_families;
string goal, alphabet;
vector<string> donors, dictionary, flowers;
int poslimit, beamwidth, familywidth, maxcost, mode;
uint32_t allowed;
array<int, 26> wanted{}, donormax{};
array<U, 26> masks{}, myersmasks{};
U low(int n) { return n >= 64 ? ~U(0) : (U(1) << n) - 1; }
U erasebits(U f, int p, int n = 1) {
  return (f & low(p)) | ((p + n >= 64 ? U(0) : f >> (p + n)) << p);
}
U insertbits(U f, int p, int n, bool frozen) {
  if (n >= 64)
    return frozen ? ~U(0) : 0;
  return (f & low(p)) | ((f & ~low(p)) << n) | (frozen ? low(n) << p : 0);
}
struct Cmd {
  int op = -1, a = 0, b = 0;
  char c = 0;
};
struct Node {
  string word;
  U frozen = 0;
  uint32_t used = 0;
  int pending = -1, g = 0, root = 0, parent = -1;
  uint32_t structure = 0;
  Cmd cmd;
};
bool avail(const Node &s, int op) {
  return (allowed & (1u << op)) && !(s.used & (1u << op));
}
bool recastable(const Node &s, int op) {
  return avail(s, RECAST) && (allowed & (1u << op));
}
string prefix(const Node &before, const Node &after) {
  auto c = after.cmd;
  string p;
  switch (c.op) {
  case ALTER:
  case CHANGE:
    p = names[c.op] + "_" + to_string(c.a) + "_" + c.c;
    break;
  case ADJ:
    p = string(c.c > before.word[c.a] ? "adj+_" : "adj-_") + to_string(c.a) +
        "_" + before.word[c.a];
    break;
  case MOVE:
    p = "move_" + to_string(c.a) + "_" + to_string(c.b);
    break;
  case ADD:
    p = "add_" + to_string(c.a) + "_" + c.c;
    break;
  case DUP:
    p = "dup_" + to_string(c.a) + "_" + before.word[c.a];
    break;
  case SUB:
    p = "sub_" + to_string(c.a) + "_" + before.word[c.a];
    break;
  case DROP:
    p = c.b ? "drop_n" : "drop_" + to_string(c.a) + "_" + before.word[c.a];
    break;
  case CHOP:
    p = "chop_" + to_string(c.a);
    break;
  case COMBINE:
    p = "combine_" + to_string(c.a) + "_" + donors[c.b];
    break;
  case SWAP:
  case SWAPADJ:
    p = string(c.op == SWAP ? "swap_" : "swap_adj_") + to_string(c.a) +
        before.word[c.a] + "_" + to_string(c.b) + before.word[c.b];
    break;
  case CHECK:
    p = "spellcheck";
    break;
  case REPLACE:
    p = string("fandr_") + char('a' + c.a) + "_" + c.c;
    break;
  case MOST:
    p = "mostf_" + to_string(c.a) + "_" + c.c;
    break;
  case UNLOCK:
    p = "unlock_" + to_string(c.a) + "_" + before.word[c.a];
    break;
  case RECAST:
    p = "recast_" + (c.a == MOST ? string("mostf") : names[c.a]);
    break;
  }
  return p + "_" + after.word;
}
int plain_distance(const string &w) {
  if (goal.empty())
    return 0;
  U pv = ~U(0), mv = 0, top = U(1) << (goal.size() - 1);
  int score = goal.size(), best = score;
  for (char c : w) {
    U eq = myersmasks[c - 'a'], xv = eq | mv, xh = (((eq & pv) + pv) ^ pv) | eq;
    U ph = mv | ~(xh | pv), mh = pv & xh;
    score += (ph & top) != 0;
    score -= (mh & top) != 0;
    ph <<= 1;
    mh <<= 1;
    pv = mh | ~(xv | ph);
    mv = ph & xv;
    best = min(best, score);
  }
  return best;
}
int distance(const string &w, U f, bool unlock, int cap) {
  if (!f)
    return min(cap + 1, plain_distance(w));
  int best = min<int>(goal.size(), cap + 1);
  if (!best)
    return 0;
  U before[65], used[65], after[65], nextused[65];
  int size = best;
  for (int d = 0; d < size; d++)
    before[d] = used[d] = low(d + 1);
  U end = U(1) << goal.size();
  for (int i = 0; i < (int)w.size(); i++) {
    U match = masks[w[i] - 'a'];
    bool locked = (f >> i) & 1;
    int nextsize = size;
    for (int d = 0; d < size; d++) {
      U a = 1 | ((before[d] << 1) & match), b = 1 | ((used[d] << 1) & match);
      if (d) {
        if (!locked) {
          a |= before[d - 1] | (before[d - 1] << 1);
          b |= used[d - 1] | (used[d - 1] << 1);
        }
        a |= after[d - 1] << 1;
        b |= nextused[d - 1] << 1;
      }
      if (unlock && locked && d >= 2)
        b |= before[d - 2] | (before[d - 2] << 1);
      if ((a | (unlock ? b : 0)) & end) {
        best = d;
        nextsize = d;
        break;
      }
      after[d] = a;
      nextused[d] = b;
    }
    if (!best)
      return 0;
    size = nextsize;
    copy(after, after + size, before);
    copy(nextused, nextused + size, used);
  }
  return best;
}
struct Alignment {
  array<uint32_t, 65> replace{}, insert{};
  U del = 0;
};
Alignment align(const Node &s) {
  int n = s.word.size(), m = goal.size(), dp[65][65]{};
  Alignment a;
  for (int j = 0; j <= m; j++)
    dp[0][j] = j;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= m; j++) {
      int cost = ((s.frozen >> (i - 1)) & 1) ? (avail(s, UNLOCK) ? 2 : 100) : 1;
      dp[i][j] =
          min({dp[i - 1][j - 1] + (s.word[i - 1] == goal[j - 1] ? 0 : cost),
               dp[i - 1][j] + cost, dp[i][j - 1] + 1});
    }
  vector<int> ends(n + 1);
  iota(ends.begin(), ends.end(), 0);
  stable_sort(ends.begin(), ends.end(),
              [&](int x, int y) { return dp[x][m] < dp[y][m]; });
  vector<pair<int, int>> todo;
  for (int k = 0; k < min(4, n + 1); k++)
    todo.emplace_back(ends[k], m);
  bool seen[65][65]{};
  while (!todo.empty()) {
    auto [i, j] = todo.back();
    todo.pop_back();
    if (!j || seen[i][j])
      continue;
    seen[i][j] = true;
    int cost =
        i && ((s.frozen >> (i - 1)) & 1) ? (avail(s, UNLOCK) ? 2 : 100) : 1;
    if (i && dp[i][j] ==
                 dp[i - 1][j - 1] + (s.word[i - 1] == goal[j - 1] ? 0 : cost)) {
      if (s.word[i - 1] != goal[j - 1])
        a.replace[i - 1] |= 1u << (goal[j - 1] - 'a');
      todo.emplace_back(i - 1, j - 1);
    }
    if (dp[i][j] == dp[i][j - 1] + 1) {
      a.insert[i] |= 1u << (goal[j - 1] - 'a');
      todo.emplace_back(i, j - 1);
    }
    if (i && dp[i][j] == dp[i - 1][j] + cost) {
      a.del |= U(1) << (i - 1);
      todo.emplace_back(i - 1, j);
    }
  }
  return a;
}
template <class Emit>
void generate(const Node &s, Emit emit, bool full = false) {
  int n = s.word.size(), visible = min(n, poslimit + 1);
  Alignment a;
  if (!full)
    a = align(s);
  array<int, 26> counts{};
  for (char c : s.word)
    counts[c - 'a']++;
  auto enabled = [&](int op) {
    return avail(s, op) && (full || s.pending < 0 || s.pending == op);
  };
  auto send = [&](int op, int i, int j, char c, string w, U f) {
    if (w.size() > 64)
      return;
    Node q;
    q.word = move(w);
    q.frozen = f;
    q.used = s.used | (1u << op);
    q.pending = -1;
    q.g = s.g + 1;
    q.root = s.root;
    q.structure = s.structure;
    if (op == COMBINE && structural_families) {
      int bucket = i == 0 ? 0 : i == n ? 5 : 1 + i * 4 / (n + 1);
      q.structure = s.structure * (donors.size() * 6 + 1) + j * 6 + bucket + 1;
    }
    q.cmd = {op, i, j, c};
    emit(move(q));
  };
  for (int i = 0; i < visible; i++)
    if (!((s.frozen >> i) & 1)) {
      char old = s.word[i];
      if (enabled(ALTER))
        for (char c : alterations[old - 'a']) {
          string w = s.word;
          w[i] = c;
          send(ALTER, i, 0, c, move(w), s.frozen | (U(1) << i));
        }
      if (enabled(ADJ))
        for (int d : {-1, 1})
          if (old + d >= 'a' && old + d <= 'z') {
            string w = s.word;
            w[i] = old + d;
            send(ADJ, i, 0, old + d, move(w), s.frozen | (U(1) << i));
          }
      if (enabled(CHANGE))
        for (char c : alphabet)
          if (!counts[c - 'a']) {
            string w = s.word;
            w[i] = c;
            send(CHANGE, i, 0, c, move(w), s.frozen | (U(1) << i));
          }
      if (enabled(SUB) && (full || ((a.del >> i) & 1))) {
        string w = s.word;
        w.erase(i, 1);
        send(SUB, i, 0, 0, move(w), erasebits(s.frozen, i));
      }
      if (enabled(DUP) &&
          (full || wanted[old - 'a'] > counts[old - 'a'] ||
           ((a.insert[i] | a.insert[i + 1]) & (1u << (old - 'a')))) &&
          n < 64) {
        string w = s.word;
        w.insert(i, 1, old);
        send(DUP, i, 0, 0, move(w), insertbits(s.frozen, i, 1, true));
      }
      if (enabled(SWAP) || enabled(SWAPADJ))
        for (int j = i + 1; j < visible; j++)
          if (!((s.frozen >> j) & 1)) {
            bool useful = full || (a.replace[i] & (1u << (s.word[j] - 'a'))) ||
                          (a.replace[j] & (1u << (old - 'a'))) ||
                          (old == s.word[j] && avail(s, REPLACE));
            if (enabled(SWAP) && useful) {
              string w = s.word;
              swap(w[i], w[j]);
              send(SWAP, i, j, 0, move(w),
                   s.frozen | (U(1) << i) | (U(1) << j));
            }
            if (enabled(SWAPADJ) && j == i + 1) {
              string w = s.word;
              swap(w[i], w[j]);
              send(SWAPADJ, i, j, 0, move(w),
                   s.frozen | (U(1) << i) | (U(1) << j));
            }
          }
      if (enabled(MOVE))
        for (int j = 0; j < visible; j++) {
          bool useful = full ||
                        ((a.insert[j] | a.replace[j] |
                          (j + 1 <= n ? a.insert[j + 1] : 0)) &
                         (1u << (old - 'a'))) ||
                        (((a.del >> i) & 1) && (j == 0 || j == visible - 1)) ||
                        (i == j && avail(s, REPLACE));
          if (!useful)
            continue;
          string w = s.word;
          w.erase(i, 1);
          w.insert(j, 1, old);
          send(MOVE, i, j, 0, move(w),
               insertbits(erasebits(s.frozen, i), j, 1, true));
        }
    }
  if (enabled(ADD) && n < 64)
    for (char c : alphabet)
      if (!counts[c - 'a'] && (full || wanted[c - 'a']))
        for (int i = 0; i <= min(n, poslimit); i++)
          if (full || (a.insert[i] & (1u << (c - 'a')))) {
            string w = s.word;
            w.insert(i, 1, c);
            send(ADD, i, 0, c, move(w), insertbits(s.frozen, i, 1, true));
          }
  if (enabled(MOST) && n < 64) {
    int c = max_element(counts.begin(), counts.end()) - counts.begin(),
        num = counts[c];
    if (num && count(counts.begin(), counts.end(), num) == 1)
      for (int i = 0; i <= min(n, poslimit); i++) {
        string w = s.word;
        w.insert(i, 1, 'a' + c);
        send(MOST, i, 0, 'a' + c, move(w), insertbits(s.frozen, i, 1, true));
      }
  }
  if (enabled(UNLOCK))
    for (int i = 0; i < visible; i++)
      if ((s.frozen >> i) & 1)
        send(UNLOCK, i, 0, 0, s.word, s.frozen & ~(U(1) << i));
  if (enabled(DROP)) {
    U done = 0;
    for (int i : {0, n - 1})
      if (i >= 0 && i < visible && !((done >> i) & 1)) {
        done |= U(1) << i;
        string w = s.word;
        w.erase(i, 1);
        send(DROP, i, 1, 0, move(w), erasebits(s.frozen, i));
      }
    for (const string &f : flowers) {
      size_t start = 0;
      while (!f.empty() && (start = s.word.find(f, start)) != string::npos) {
        for (int i : {int(start), int(start + f.size() - 1)})
          if (i < visible && !((done >> i) & 1)) {
            done |= U(1) << i;
            string w = s.word;
            w.erase(i, 1);
            send(DROP, i, 0, 0, move(w), erasebits(s.frozen, i));
          }
        start++;
      }
    }
  }
  if (enabled(CHOP))
    for (int side = 0; side < 2; side++) {
      int start = side ? n / 2 : 0, len = (n + 1) / 2;
      send(CHOP, side, 0, 0, s.word.substr(start, len),
           (s.frozen >> start) & low(len));
    }
  if (enabled(COMBINE))
    for (int k = 0; k < (int)donors.size(); k++)
      if (n + donors[k].size() <= 64)
        for (int i = 0; i <= min(n, poslimit); i++) {
          if (!full && s.g > 0 && i != 0 && i != n && !a.insert[i] &&
              !a.replace[i])
            continue;
          string w = s.word;
          w.insert(i, donors[k]);
          send(COMBINE, i, k, 0, move(w),
               insertbits(s.frozen, i, donors[k].size(), false));
        }
  if (enabled(CHECK)) {
    string w = s.word;
    U f = s.frozen;
    for (const string &d : dictionary) {
      size_t p = 0;
      while ((p = w.find(d, p)) != string::npos) {
        w.erase(p, d.size());
        f = erasebits(f, p, d.size());
      }
    }
    send(CHECK, 0, 0, 0, move(w), f);
  }
  if (enabled(REPLACE))
    for (int from = 0; from < 26; from++)
      if (counts[from])
        for (char to : alphabet)
          if (!counts[to - 'a'] && (full || wanted[to - 'a'])) {
            string w = s.word;
            U f = s.frozen;
            for (int i = 0; i < n; i++)
              if (w[i] == 'a' + from && !((f >> i) & 1)) {
                w[i] = to;
                f |= U(1) << i;
              }
            send(REPLACE, from, 0, to, move(w), f);
          }
  if (enabled(RECAST))
    for (int op = 0; op < 17; op++)
      if (op != RECAST && op != UNLOCK && (allowed & (1u << op)) &&
          (s.used & (1u << op))) {
        Node q = s;
        q.used = (q.used | (1u << RECAST)) & ~(1u << op);
        q.pending = op;
        q.g++;
        q.cmd = {RECAST, op, 0, 0};
        emit(move(q));
      }
}
U fingerprint(const Node &s) {
  U h = 1469598103934665603ULL;
  for (unsigned char c : s.word)
    h = (h ^ c) * 1099511628211ULL;
  h ^= s.frozen + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
  h ^= U(s.used) + (U(s.pending + 1) << 24);
  return h;
}
bool feasible(const Node &s, const array<int, 26> &counts,
              const array<int, 26> &unfrozen) {
  bool comb = avail(s, COMBINE) || recastable(s, COMBINE),
       fr = avail(s, REPLACE) || recastable(s, REPLACE);
  int creators = avail(s, ADD) + avail(s, CHANGE) + avail(s, ALTER) +
                 avail(s, ADJ) + avail(s, REPLACE);
  if (avail(s, RECAST))
    creators++;
  int types = 0;
  for (int c = 0; c < 26; c++)
    if (wanted[c] && !counts[c] && (!comb || !donormax[c]))
      types++;
  if (types > creators)
    return false;
  for (int c = 0; c < 26; c++)
    if (wanted[c] >= 5) {
      bool rec = avail(s, RECAST);
      int combos =
          avail(s, COMBINE) + (rec && (allowed & (1u << COMBINE)) ? 1 : 0);
      uint32_t preserving =
                   (1u << DUP) | (1u << MOST) | (1u << ALTER) | (1u << ADJ),
               growth = preserving | (1u << ADD) | (1u << CHANGE);
      int p = __builtin_popcount((allowed & ~s.used) & preserving) +
              (rec && (allowed & preserving) ? 1 : 0);
      int all = __builtin_popcount((allowed & ~s.used) & growth) +
                (rec && (allowed & growth) ? 1 : 0);
      int supplied = combos * donormax[c],
          maximum = max(counts[c] + supplied + p, supplied + all);
      if (fr) {
        int batch = 0;
        for (int x = 0; x < 26; x++)
          if (x != c)
            batch = max(batch, unfrozen[x] + combos * donormax[x]);
        maximum = max(maximum, batch + int(avail(s, UNLOCK)) + supplied + p);
      }
      if (wanted[c] > maximum)
        return false;
    }
  if (!comb && !fr && !avail(s, RECAST)) {
    int preserving =
            avail(s, DUP) + avail(s, MOST) + avail(s, ALTER) + avail(s, ADJ),
        growth = preserving + avail(s, ADD) + avail(s, CHANGE), missing = 0;
    for (int c = 0; c < 26; c++) {
      missing += max(0, wanted[c] - counts[c]);
      if (wanted[c] > max(counts[c] + preserving, growth))
        return false;
    }
    if (missing > growth)
      return false;
  }
  return true;
}
float rank_state(const Node &s, float cutoff) {
  array<int, 26> counts{}, unfrozen{};
  for (int i = 0; i < (int)s.word.size(); i++) {
    int c = s.word[i] - 'a';
    counts[c]++;
    if (!((s.frozen >> i) & 1))
      unfrozen[c]++;
  }
  if (!feasible(s, counts, unfrozen))
    return 1e6;
  auto missing = [&](const array<int, 26> &cs) {
    int n = 0;
    for (int c = 0; c < 26; c++)
      n += max(0, wanted[c] - cs[c]);
    return .15f * n;
  };
  float penalty = missing(counts), soft = plain_distance(s.word) + penalty;
  vector<pair<string, float>> previews;
  if (avail(s, REPLACE)) {
    array<int, 26> sorted;
    iota(sorted.begin(), sorted.end(), 0);
    partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(),
                 [&](int a, int b) { return unfrozen[a] > unfrozen[b]; });
    for (int c = 0; c < 26; c++)
      if (wanted[c] >= 3 && !counts[c])
        for (int k = 0; k < 2; k++) {
          int from = sorted[k];
          if (unfrozen[from] < 3)
            continue;
          string w = s.word;
          for (int i = 0; i < (int)w.size(); i++)
            if (w[i] == 'a' + from && !((s.frozen >> i) & 1))
              w[i] = 'a' + c;
          auto cs = counts;
          cs[c] += unfrozen[from];
          cs[from] -= unfrozen[from];
          float p = 1 + missing(cs);
          soft = min(soft, plain_distance(w) + p);
          previews.emplace_back(move(w), p);
        }
  }
  if (soft > cutoff)
    return 1e6;
  int cap = min<int>(goal.size(), floor(cutoff - penalty));
  if (cap < 0)
    cap = 0;
  float result = distance(s.word, s.frozen, avail(s, UNLOCK), cap) + penalty;
  for (auto &[w, p] : previews)
    result =
        min(result, distance(w, s.frozen, avail(s, UNLOCK), goal.size()) + p);
  return result;
}
struct Entry {
  float rank;
  U serial;
  int index;
  bool operator<(const Entry &o) const {
    return tie(rank, serial) < tie(o.rank, o.serial);
  }
};
int main() {
  ios::sync_with_stdio(false);
  cin.tie(nullptr);
  int ns, nd, nf, nr;
  string protocol;
  cin >> protocol;
  if (protocol != "FLEURDREAM_NATIVE_1") {
    cerr << "Unsupported native discovery protocol\n";
    return 2;
  }
  cin >> goal >> alphabet >> poslimit >> allowed >> beamwidth >> familywidth >>
      maxcost >> mode >> structural_families;
  if (goal == "-")
    goal = "";
  if (goal.size() > 63)
    return 2;
  for (int i = 0; i < (int)goal.size(); i++) {
    wanted[goal[i] - 'a']++;
    masks[goal[i] - 'a'] |= U(1) << (i + 1);
    myersmasks[goal[i] - 'a'] |= U(1) << i;
  }
  cin >> ns;
  for (int k = 0; k < ns; k++) {
    string w;
    cin >> w;
    donors.push_back(w);
    array<int, 26> c{};
    for (char x : w)
      c[x - 'a']++;
    for (int x = 0; x < 26; x++)
      donormax[x] = max(donormax[x], c[x]);
  }
  cin >> nd;
  for (int k = 0; k < nd; k++) {
    string w;
    cin >> w;
    dictionary.push_back(w);
  }
  cin >> nf;
  for (int k = 0; k < nf; k++) {
    string w;
    cin >> w;
    flowers.push_back(w);
  }
  for (string &mapping : alterations) {
    cin >> mapping;
    if (mapping == "-")
      mapping = "";
  }
  cin >> nr;
  vector<Node> roots;
  for (int i = 0; i < nr; i++) {
    Node s;
    cin >> s.word >> s.frozen >> s.used >> s.g >> s.pending;
    s.root = i;
    if (s.word == "-")
      s.word = "";
    roots.push_back(s);
  }
  if (!cin)
    return 2;
  if (mode == 1) {
    for (const auto &s : roots)
      generate(
          s,
          [&](Node q) {
            cout << (q.word.empty() ? "-" : q.word) << ' ' << q.frozen << ' '
                 << q.used << ' ' << q.pending << ' ' << prefix(s, q) << '\n';
          },
          true);
    return 0;
  }
  if (mode == 2) {
    for (const auto &s : roots)
      cout << plain_distance(s.word) << ' '
           << distance(s.word, s.frozen, true, goal.size()) << '\n';
    return 0;
  }
  int capacity = beamwidth * 4;
  vector<Node> arena;
  arena.reserve(size_t(maxcost + 2) * capacity + roots.size() + 1);
  struct Bucket {
    priority_queue<Entry> heap;
    int live = 0;
  };
  struct Layer {
    priority_queue<Entry> heap;
    unordered_map<U, Bucket> buckets;
    int live = 0;
  };
  vector<Layer> layers(maxcost + 1);
  vector<unordered_set<U>> seen(maxcost + 1);
  vector<U> versions;
  versions.reserve(arena.capacity());
  U serial = 0, generated = 0, expanded = 0;
  int winner = -1;
  auto clean = [&](priority_queue<Entry> &h) {
    while (!h.empty() && versions[h.top().index] != h.top().serial)
      h.pop();
  };
  auto put = [&](Node s, int parent) {
    generated++;
    if (winner >= 0 || s.g > maxcost)
      return;
    if (parent >= 0) {
      const Node &b = arena[parent];
      if (b.pending < 0 && b.word == s.word && b.frozen == s.frozen &&
          (b.used & s.used) == b.used)
        return;
    }
    s.parent = parent;
    if (s.word.find(goal) != string::npos) {
      winner = arena.size();
      arena.push_back(move(s));
      return;
    }
    U key = fingerprint(s);
    auto &keys = seen[s.g];
    if (!keys.insert(key).second)
      return;
    auto &layer = layers[s.g];
    U fam = (U(s.root) << 40) | (U(s.structure) << 17) | s.used;
    auto &bucket = layer.buckets[fam];
    clean(layer.heap);
    clean(bucket.heap);
    float cutoff = 1e5;
    if (layer.live == capacity)
      cutoff = layer.heap.top().rank;
    if (bucket.live == familywidth)
      cutoff = min(cutoff, bucket.heap.top().rank);
    float rank = rank_state(s, cutoff);
    if (rank > cutoff || rank >= 1e5)
      return;
    int index = -1;
    if (bucket.live == familywidth) {
      if (rank >= bucket.heap.top().rank)
        return;
      index = bucket.heap.top().index;
      bucket.heap.pop();
      bucket.live--;
      layer.live--;
      versions[index] = 0;
    } else if (layer.live == capacity) {
      if (rank >= layer.heap.top().rank)
        return;
      index = layer.heap.top().index;
      layer.heap.pop();
      const Node &old = arena[index];
      U oldfam = (U(old.root) << 40) | (U(old.structure) << 17) | old.used;
      layer.buckets[oldfam].live--;
      layer.live--;
      versions[index] = 0;
    }
    U version = ++serial;
    if (index < 0) {
      index = arena.size();
      arena.push_back(move(s));
      versions.push_back(version);
    } else {
      arena[index] = move(s);
      versions[index] = version;
    }
    Entry entry = {rank, version, index};
    layer.heap.push(entry);
    bucket.heap.push(entry);
    layer.live++;
    bucket.live++;
  };
  for (Node s : roots)
    put(move(s), -1);
  for (int depth = 0; depth <= maxcost && winner < 0; depth++) {
    vector<Entry> group;
    auto &layer = layers[depth];
    auto &heap = layer.heap;
    while (!heap.empty()) {
      auto e = heap.top();
      heap.pop();
      if (versions[e.index] == e.serial)
        group.push_back(e);
    }
    reverse(group.begin(), group.end());
    layer.buckets.clear();
    layer.live = 0;
    unordered_map<U, int> families;
    vector<int> chosen;
    for (auto e : group) {
      const Node &s = arena[e.index];
      U family = (U(s.root) << 40) | (U(s.structure) << 17) | s.used;
      if (families[family]++ >= familywidth)
        continue;
      chosen.push_back(e.index);
      if (chosen.size() == (size_t)beamwidth)
        break;
    }
    seen[depth].clear();
    seen[depth].rehash(0);
    cerr << "layer " << depth << " frontier " << group.size() << " expanded "
         << chosen.size() << " total " << expanded << " generated " << generated
         << '\n';
    for (int index : chosen) {
      Node s = arena[index];
      expanded++;
      generate(s, [&](Node q) { put(move(q), index); });
      if (winner >= 0)
        break;
    }
  }
  if (winner < 0) {
    cout << "MISS " << expanded << ' ' << generated << '\n';
    return 0;
  }
  vector<int> path;
  for (int i = winner; arena[i].parent >= 0; i = arena[i].parent)
    path.push_back(i);
  reverse(path.begin(), path.end());
  cout << "FOUND " << arena[winner].root << ' ' << arena[winner].g << ' '
       << expanded << ' ' << generated << '\n';
  for (int i : path)
    cout << prefix(arena[arena[i].parent], arena[i]) << '\n';
}
