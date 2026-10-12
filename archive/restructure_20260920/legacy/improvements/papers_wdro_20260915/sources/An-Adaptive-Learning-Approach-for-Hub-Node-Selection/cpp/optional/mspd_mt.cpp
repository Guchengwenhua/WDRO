// Optional Boost-based batch wrapper for the MSPD implementation.
// See third_party/MSPD-BSD-3-Clause.txt for the upstream license.
#include "mspd_mt.hpp"
#include <cassert>
#define orig_num_nets (*orig_num_nets_Pointer)
#define X X_Pointer.get()
#define Y Y_Pointer.get()
#define key key_Pointer.get()
#define PL PL_Pointer.get()
#define par par_Pointer.get()
#define parCnt parCnt_Pointer.get()
#define visited visited_Pointer.get()
#define cnt cnt_Pointer.get()
#define DT DT_Pointer.get()
#define nWL (*nWL_Pointer)
#define nSkew (*nSkew_Pointer)
#define points (*points_Pointer)
#define adj adj_Pointer.get()
#define subTree subTree_Pointer.get()
#define nn nn_Pointer.get()
#define sN sN_Pointer.get()

void STT::initThreadSpecificData() {
  orig_num_nets_Pointer.reset(new int(0));
  X_Pointer.reset(new int[MX]);
  Y_Pointer.reset(new int[MX]);
  key_Pointer.reset(new int[MX]);
  PL_Pointer.reset(new int[MX]);
  par_Pointer.reset(new int[MX]);
  parCnt_Pointer.reset(new int[MX]);
  visited_Pointer.reset(new int[MX]);
  cnt_Pointer.reset(new int[MX]);
  DT_Pointer.reset(new int[MX]);
  nWL_Pointer.reset(new int(0));
  nSkew_Pointer.reset(new int(0));
  points_Pointer.reset(new std::vector<std::pair<std::pair<int, int>, int>>());
  adj_Pointer.reset(new std::set<int>[MX]);
  subTree_Pointer.reset(new std::set<int>[MX]);
  nn_Pointer.reset(new std::vector<int>[MX]);
  sN_Pointer.reset(new std::vector<int>[MX]);
}
vector<pair<float, float>> STT::solve_single(const vector<int>& arr, const vector<int>& source, bool output) {
  orig_num_nets = arr.size() / 2;
  for (int i = 0; i < orig_num_nets; i++) {
    X[i] = arr[i * 2], Y[i] = arr[i * 2 + 1];
    nn[i].clear();
  }
  vector<pair<float, float>> ans;
  for (float alpha = 0.0; alpha <=1.0; alpha+=0.1) {
    enterInput();
    enterInput();
    getNearestNeighbors();
    PrimDijktra(alpha, source, 1);
    auto random_WL = calcWL();
    auto random_skew = calcSkew();
    if (output) {    
      cout<<"random_WL:"<<random_WL<<endl;
      cout<<"random_skew:"<<random_skew<<endl;
    }
    ans.push_back({random_WL, random_skew});
  }
  return ans;
}
template <typename T>
vector<pair<float, float>> STT::solve_batch(const vector<vector<int>>& arr, const vector<vector<int>>& sources, T obj, bool output){
    int batch = arr.size();
    assert (batch == sources.size());
    float w1 = 0.5, w2 = 0.5;
    if (is_same<T, int>::value) {
        if (obj == 1)
            {w1 = 0.5;
            w2=0.5 ;}
        else if (obj == 2)
            {w1 = 0.25;
            w2=0.75 ;}
        else if (obj == 3)
            {w1=0.75;
            w2=0.25;}
        else {
            cerr<<"No other integer object (only 1 ,2,3), please specify float weight directly."<<endl;
            exit(0);
        }
    }
    else if(is_same<T, float>::value) {
        assert(obj <= 1 && obj >= 0);
        w1 = obj;
        w2 = 1-obj;
    }
    else {
        cerr<<"No other type supported for obj parameter (only int and float)."<<endl;
        exit(0);
    }    
    vector<pair<float, float>> ret(batch);
    for (int i=0; i<batch; i++) {
        auto worker = [this, i, arr = arr[i], source = sources[i], w1, w2, &ret]() {
            this->thread_worker(arr, source, w1, w2, i, ret);
        };
        boost::asio::post(pool, worker);
    }
    pool.join();
    return ret;
}

void STT::thread_worker(const vector<int>& arr, const vector<int>& source, float w1, float w2, int id, vector<pair<float, float>>& ret){
    initThreadSpecificData();
    auto&& ans = solve_single(arr, source);
    int minObj = 1e9;
    int minWL = 1e9;
    int minSkew = 1e9;
    for (auto& [wl, skew] : ans){
        if (w1 * wl + w2 * skew < minObj) {
            minObj = w1 * wl + w2 * skew ;
            minWL = wl;
            minSkew = skew;
        }
    }
    ret[id] = make_pair(minWL, minSkew);
}

int STT::getDist(int i, int j) { return abs(X[i] - X[j]) + abs(Y[i] - Y[j]); }
void STT::enterInput() {
  // erase everything before

  for (int i = 0; i < sz(points); i++) {
    sN[i].clear();
    adj[i].clear(), nn[i].clear(), subTree[i].clear();
    key[i] = PL[i] = INF;
    par[i] = visited[i] = cnt[i] = parCnt[i] = DT[i] = 0;
  }
  points.clear();
  for (int i = 0; i < orig_num_nets; i++)
    points.pb(mp(mp(X[i], Y[i]), i));
}

int STT::calcWL() {
    int sum = 0;
    for (int i = 0; i < sz(points); i++) {
      for (auto x : adj[i])
        sum += getDist(i, x);
    }
  return sum;
}

void STT::dfs(int at, int dist) {
  subTree[at].insert(at);
  PL[at] = dist, parCnt[at]++;
  cnt[at]++;
  for (auto x : adj[at])
    if (x != at) {
      dfs(x, dist + getDist(at, x));
      for (auto v : subTree[x])
        subTree[at].insert(v);
      cnt[at] += cnt[x];
    }
}

int STT::calcSkew() {
  for (int i = 0; i < sz(points); i++) {
    cnt[i] = PL[i] = parCnt[i] = 0;
    subTree[i].clear();
  }
  dfs(0, 0);

  int mx = 0, mn = 1000000000;
  for (int i = 1; i < orig_num_nets; i++) {
    mx = max(mx, PL[i]), mn = min(mn, PL[i]);
  }

  return mx - mn;
}

void  STT::getDT(int at, int p) {
  DT[at] = PL[at] - getDist(at, 0);
  for (auto x : adj[at])
    if (x != p) {
      getDT(x, at);
    }
}

vector<int>  STT::getParents() {
  vector<int> ret;
  for (int i = 0; i < sz(points); i++) {
    ret.pb(par[i]);
    ret.pb(X[i]);
    ret.pb(Y[i]);
  }
  return ret;
}

void  STT::getNearestNeighbors() {
  int nSize = sz(points);
  vector<int> ur(nSize, numeric_limits<int>::max());
  vector<int> ul(nSize, numeric_limits<int>::min());
  vector<int> lr(nSize, numeric_limits<int>::max());
  vector<int> ll(nSize, numeric_limits<int>::min());

  // sort by y-coordinate
  vector<pair<pii, int>> sorted;
  // (y, x, idx)
  for (int i = 0; i < sz(points); i++)
    sorted.pb(mp(mp(points[i].f.s, points[i].f.f), points[i].s));
  for (int i = 0; i < sz(points); i++)
    nn[i].clear();
  sort(begin(sorted), end(sorted));

  for (int i = 0; i < sz(points); i++) {

    int a = sorted[i].s, x1 = sorted[i].f.s;

    for (int j = 0; j < i; j++) {
      int b = sorted[j].s, x2 = sorted[j].f.s;
      if (x2 <= x1 && x1 < ur[b]) {
        ur[b] = x1;
        nn[b].pb(a);
      } else if (ul[b] < x1 && x1 < x2) {
        ul[b] = x1;
        nn[b].pb(a);
      }
    }

    for (int j = i - 1; j >= 0; j--) {
      int b = sorted[j].s, x2 = sorted[j].f.s;
      if (x1 <= x2 && x2 < lr[a]) {
        lr[a] = x2;
        nn[a].pb(b);
      } else if (ll[a] < x2 && x2 < x1) {
        ll[a] = x2;
        nn[a].pb(b);
      }
    }
  }
}

vector<int>  STT::bestSteinerNode(int idx) {
  int v = idx, a = 0, b = 0, x = X[idx], y = Y[idx], best = 0;
  int bX = 0, bY = 0;

  for (auto A : adj[idx])
    if (A != idx) {

      for (auto B : adj[idx])
        if (B != idx && A != B) {

          int sX = X[idx], sY = Y[idx];

          if (min(X[A], X[B]) > X[idx]) {
            sX = min(X[A], X[B]);
          }

          else if (max(X[A], X[B]) < X[idx]) {
            sX = max(X[A], X[B]);
          }

          if (min(Y[A], Y[B]) > Y[idx]) {
            sY = min(Y[A], Y[B]);
          } else if (max(Y[A], Y[B]) < Y[idx]) {
            sY = max(Y[A], Y[B]);
          }

          int gain = abs(X[idx] - sX) + abs(Y[idx] - sY);
          if (gain > best) {
            best = gain, v = idx, a = A, b = B, bX = sX, bY = sY;
          }
        }
    }

  vector<int> ret;

  ret.pb(best);
  // par, child, child
  ret.pb(v);
  ret.pb(a);
  ret.pb(b);
  // x, y of new steiner
  ret.pb(bX);
  ret.pb(bY);
  return ret;
}

void  STT::steinerize() {
  set<vector<int>> pq;

  for (int i = 1; i < sz(points); i++) {
    vector<int> newNode = bestSteinerNode(i);
    pq.insert(newNode);
    sN[i] = newNode;
  }

  while (sz(pq)) {
    vector<int> cur = *pq.rbegin();
    pq.erase(cur);
    if (cur[0] == 0)
      break;

    int a = cur[2], b = cur[3];
    int steinerNode = sz(points);

    if (X[a] == cur[4] && Y[a] == cur[5])
      steinerNode = a;
    else if (X[b] == cur[4] && Y[b] == cur[5])
      steinerNode = b;
    else {
      points.pb(mp(mp(cur[4], cur[5]), steinerNode));
      X[steinerNode] = cur[4], Y[steinerNode] = cur[5];
    }

    if (a != steinerNode) {
      adj[cur[1]].erase(a);
      adj[cur[1]].insert(steinerNode);
      par[steinerNode] = cur[1];
      adj[steinerNode].insert(a);
      par[a] = steinerNode;
      par[steinerNode] = cur[1];
    }

    if (b != steinerNode) {
      adj[cur[1]].erase(b);
      adj[cur[1]].insert(steinerNode);
      par[steinerNode] = cur[1];
      adj[steinerNode].insert(b);
      par[b] = steinerNode;
      par[steinerNode] = cur[1];
    }

    vector<int> nS = bestSteinerNode(cur[1]);
    sN[cur[1]] = nS;
    pq.insert(nS);

    if (pq.count(sN[a]))
      pq.erase(sN[a]);
    if (pq.count(sN[b]))
      pq.erase(sN[b]);

    nS = bestSteinerNode(a);
    pq.insert(nS);
    sN[a] = nS;

    nS = bestSteinerNode(b);
    pq.insert(nS);
    sN[b] = nS;
  }
}
bool contains(const vector<int>& vec, int value) {
    return find(vec.begin(), vec.end(), value) != vec.end();
}
void  STT::DAS(const vector<int>& s) {
  getNearestNeighbors();

  for (int i = 0; i < sz(points); i++)
    subTree[i].clear();
  for (int i = 0; i < sz(points); i++)
    parCnt[i] = cnt[i] = PL[i] = 0;
  dfs(0, 0);

  int mxPL = 0;
  for (int i = 0; i < orig_num_nets; i++)
    mxPL = max(mxPL, PL[i]);

  for (int i = 1; i < sz(points); i++)
    if (!contains(s, i)) { // i is not source
      int edgeLen = getDist(par[i], i);
      int bestNN = par[i];

      if (bestNN >= orig_num_nets)
        continue;

      for (auto x : nn[i])
        if (x != 0) {
          if (getDist(x, i) <= edgeLen && PL[i] <= 0.5 * mxPL) {
            if (!subTree[i].count(x)) {
              bestNN = x;
              edgeLen = getDist(x, i);
            }
          }
        }

      adj[par[i]].erase(i);
      par[i] = bestNN;
      adj[par[i]].insert(i);

      for (int j = 0; j < sz(points); j++)
        subTree[j].clear();
      for (int j = 0; j < sz(points); j++)
        PL[j] = parCnt[j] = cnt[j] = 0;
      dfs(0, 0);
    }

  for (int i = 0; i < sz(points); i++) {
    subTree[i].clear();
    parCnt[i] = cnt[i] = PL[i] = 0;
  }

  dfs(0, 0);
  getDT(0, 0);

  mxPL = 0;
  int curDT = 0, curWL = calcWL();
  for (int i = 0; i < orig_num_nets; i++)
    mxPL = max(mxPL, PL[i]);
  for (int i = 0; i < sz(points); i++)
    curDT += DT[i];

  for (int i = 1; i < sz(points); i++)
    if (!contains(s, i)) {
      int newPar = par[i];

      if (newPar >= orig_num_nets)
        continue;

      for (auto x : nn[i])
        if (x != 0) {
          int newWL = curWL - getDist(i, par[i]) + getDist(i, x);
          int newDT =
              curDT - (getDist(i, par[i]) * cnt[i]) + (getDist(i, x) * cnt[i]);
          if (newWL <= curWL && newDT <= curDT) {
            if (!subTree[i].count(x)) {
              curWL = newWL, curDT = newDT;
              newPar = x;
            }
          }
        }

      adj[par[i]].erase(i);
      par[i] = newPar;
      adj[par[i]].insert(i);

      for (int j = 0; j < sz(points); j++) {
        subTree[j].clear();
        parCnt[j] = cnt[j] = PL[j] = 0;
      }
      dfs(0, 0);
      getDT(0, 0);
    }

  for (int i = 0; i < sz(points); i++) {
    subTree[i].clear();
    parCnt[i] = cnt[i] = PL[i] = 0;
  }
  dfs(0, 0);

  for (int i = 0; i < sz(points); i++)
    if (parCnt[i] != 1)
      cout << "error\n";
}

void STT::PrimDijktra(float alp, const vector<int>& s, int T) {

  // weight (w. alpha), pathlength, idx, par_idx
  set<pair<pii, pii>> pq;
  pq.insert(mp(mp(0, 0), mp(0, 0)));
  key[0] = PL[0] = par[0] = 0;
  for (auto x : s)
    if (x != 0) {
      key[x] = PL[x] = par[x] = 0;
      pq.insert(mp(mp(key[x], PL[x]), mp(x, 0)));
    }

  if (sz(pq) > 1) {
    visited[0] = 1;
    pq.erase(mp(mp(key[0], PL[0]), mp(0, 0)));
  }

  while (sz(pq)) {

    pair<pii, pii> fr = *pq.begin();
    pq.erase(pq.begin());
    visited[fr.s.f] = 1;

    for (auto x : nn[fr.s.f]) {
      int len = getDist(fr.s.f, x), pl = len + fr.f.s,
          weight = alp * fr.f.s + len;
      if (!visited[x] && weight <= key[x]) {
        if (pq.count(mp(mp(key[x], PL[x]), mp(x, par[x]))))
          pq.erase(mp(mp(key[x], PL[x]), mp(x, par[x])));
        key[x] = weight;
        PL[x] = pl;
        par[x] = fr.s.f;
        pq.insert(mp(mp(key[x], PL[x]), mp(x, par[x])));
      }
    }
  }

  for (int i = 0; i < orig_num_nets; i++)
    adj[par[i]].insert(i);

  if (T) {
    steinerize();
    DAS(s);
  }
}

// class test{
// public:
//   test(){}
//   void print(){std::cout<<"NNNNNNNNN"<<std::endl;}
// };
// void ttt(){
//   std::cout<<"YYYYYYYY"<<std::endl;
// }
PYBIND11_MODULE( _mspd_mt, m ){
    m.doc() = "mspd for STT with multi-thread.";
    // m.def("ttt", &ttt, "get Tree" );
    pybind11::class_<STT>(m, "STT")
        .def(pybind11::init<int>())
        .def(pybind11::init<>())
        .def("solve_batch", &STT::solve_batch<int>)
        .def("solve_batch", &STT::solve_batch<float>);
}
// Requires pybind11, Boost.Thread, Boost.System, pthreads, and C++17.

// int main(){
//   cout<<"trest"<<endl;
//   STT* stt = new STT(10);
//   cout<<"trest222"<<endl;
//   vector<vector<int>> arr = {{1,2,3,4,5,6},{7,8,9,1,2,3},{7,8,9,1,2,3},{7,8,9,1,2,3},{7,8,9,1,2,3},{7,8,9,1,2,3}};
//   vector<vector<int>> s = {{1,2}, {1,0},{1,0},{1,0},{1,0},{1,0}};
//   auto && ans = stt->solve_batch(arr, s, 1);
//    cout<<"trest333"<<endl;
//   for (auto&[w,s] : ans) {
//     cout<<w<<" "<<s<<endl;
//   }
//   return 0;
// }
