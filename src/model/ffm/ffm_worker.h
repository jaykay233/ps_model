/*
* ffm_worker.h
* @author: jaykay233
*/

#ifndef SRC_MODEL_FFM_FFM_WORKER_H_
#define SRC_MODEL_FFM_FFM_WORKER_H_

#include "src/io/load_data_from_disk.h"
#include "src/base/thread_pool.h"
#include "src/base/base.h"
#include "ps/ps.h"

namespace xflow{
    class FFMWorker(const char *train_file,const char *test_file) :
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
};


#endif


