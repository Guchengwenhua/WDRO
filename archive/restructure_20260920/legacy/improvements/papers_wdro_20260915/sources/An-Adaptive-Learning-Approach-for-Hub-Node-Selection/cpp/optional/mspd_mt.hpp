#include <boost/thread.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <queue>
#include <set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <boost/thread/tss.hpp>
#include <boost/asio.hpp>

using namespace std;
const int MX = 1e5 + 5;
const int INF = 1e9;

typedef pair<int, int> pii;

#define f first
#define s second
#define mp make_pair
#define pb push_back
#define sz(x) (int)x.size()

void array_deleter(int* arr) {
    delete[] arr;
}

void set_deleter(set<int>* arr) {
    delete[] arr;
}

void vector_deleter(vector<int>* arr) {
    delete[] arr;
}

class STT{
public:
  STT(int num_thread) : pool(num_thread) {}
  STT() : pool(10) {}
  vector<pair<float, float>> solve_single(const vector<int> &arr, const vector<int>& s, bool output=false);
  // vector<pair<float, float>> solve_batch(const vector<vector<int>>& arr, const vector<vector<int>>& s, T obj, int num_threads, bool output=false);
  template <typename T>
  vector<pair<float, float>> solve_batch(const vector<vector<int>>& arr, const vector<vector<int>>& s, T obj, bool output=false);

private:
  void thread_worker(const vector<int>& arr, const vector<int>& s, float w1, float w2, int id, vector<pair<float, float>>& ret);

private:
  void enterInput();
  int getDist(int i, int j);
  int calcWL();

  void dfs(int at, int dist) ;
  int calcSkew() ;

  void getDT(int at, int par) ;
  vector<int> getParents() ;

  void getNearestNeighbors() ;

  vector<int> bestSteinerNode(int idx);
  void steinerize() ;
  void DAS(const vector<int>& s) ;
  void PrimDijktra(float alp, const vector<int>& s, int T) ;

private:
    // int orig_num_nets, X[MX], Y[MX];
    // int key[MX], PL[MX], par[MX];
    // int parCnt[MX], visited[MX], cnt[MX], DT[MX];
    // int nWL, nSkew;
    // vector<pair<pii, int>> points;
    // set<int> adj[MX], subTree[MX];
    // vector<int> nn[MX], sN[MX];
    boost::asio::thread_pool pool;
    
    void initThreadSpecificData();
    boost::thread_specific_ptr<int> orig_num_nets_Pointer;
    boost::thread_specific_ptr<int> X_Pointer{array_deleter};
    boost::thread_specific_ptr<int> Y_Pointer{array_deleter};
    boost::thread_specific_ptr<int> key_Pointer{array_deleter};
    boost::thread_specific_ptr<int> PL_Pointer{array_deleter};
    boost::thread_specific_ptr<int> par_Pointer{array_deleter};
    boost::thread_specific_ptr<int> parCnt_Pointer{array_deleter};
    boost::thread_specific_ptr<int> visited_Pointer{array_deleter};
    boost::thread_specific_ptr<int> cnt_Pointer{array_deleter};
    boost::thread_specific_ptr<int> DT_Pointer{array_deleter};
    boost::thread_specific_ptr<int> nWL_Pointer{array_deleter};
    boost::thread_specific_ptr<int> nSkew_Pointer{array_deleter};
    boost::thread_specific_ptr<std::vector<std::pair<pii, int>>> points_Pointer;
    boost::thread_specific_ptr<std::set<int>> adj_Pointer{set_deleter};
    boost::thread_specific_ptr<std::set<int>> subTree_Pointer{set_deleter};
    boost::thread_specific_ptr<std::vector<int>> nn_Pointer{vector_deleter};
    boost::thread_specific_ptr<std::vector<int>> sN_Pointer{vector_deleter};

};

