/*
* ffm_worker.h
* @author: jaykay233
*/

#ifndef SRC_MODEL_FFM_FFM_WORKER_H_
#define SRC_MODEL_FFM_FFM_WORKER_H_

#include <time.h>
#include <unistd.h>
#include <immintrin.h>

#include <algorithm>
#include <vector>
#include <ctime>
#include <iostream>
#include <mutex>
#include <functional>
#include <random>
#include <string>
#include <memory>

#include "src/io/load_data_from_disk.h"
#include "src/base/thread_pool.h"
#include "src/base/base.h"
#include "ps/ps.h"
#include "src/io/io.h"

namespace xflow{
class FFMWorker{
    public:
        FFMWorker(const char *train_file,const char *test_file) :
            train_file_path(train_file),
            test_file_path(test_file){
                kv_w = new ps::KVWorker<float>(0);
                kv_v = new ps::KVWorker<float>(1);
                base_ = new Base;
                core_num = std::thread::hardware_concurrency();
                pool_ = new ThreadPool(core_num);
            }

        ~FFMWorker() {}

        void batch_training(ThreadPool* pool);
        void train();
        void update(int start,int end);
        void predict(ThreadPool* pool_,int rank,int block);
        void calculate_pctr(int start,int end);
        void calculate_loss(std::vector<float>& w,std::vector<float>& v,std::vector<Base::sample_key>& all_keys,std::vector<ps::Key>& unique_keys,size_t start,size_t end,std::vector<float>& v_sum,std::vector<float>& loss);
        void calculate_gradient(std::vector<Base::sample_key>& all_keys,std::vector<ps::Key>& unique_keys,size_t start,size_t end,std::vector<float>& v,std::vector<float>& v_sum,std::vector<float>& loss,std::vector<float>& push_w_gradient,std::vector<float>& push_v_gradient);
        int epochs = 60;
        private:
            int rank;
            int core_num;
            int block_size = 2;

            std::atomic_llong num_batch_fly = {0};
            std::atomic_llong gradient_thread_finish_num = {0};
            std::atomic_llong calculate_pctr_thread_finish_num = {0};

            std::vector<Base::auc_key> auc_vec;
            std::vector<Base::auc_key> test_auc_vec;

            std::ofstream md;
            std::mutex mutex;
            Base* base_;
            ThreadPool* pool_;
            xflow::Data *train_data;
            xflow::Data *test_data;
            const char *train_file_path;
            const char *test_file_path;
            char train_data_path[1024];
            char test_data_path[1024];
            int v_dim_ = 20;
            int f_dim_ = 20;
            int e_dim_ = 1;
            ps::KVWorker<float>* kv_w;
            ps::KVWorker<float>* kv_v;
    };
};


#endif


