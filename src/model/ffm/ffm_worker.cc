/*
ffm_worker.cc
@author: zhiyuanxu
*/

#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include <utility>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <mutex>
#include <functional>
#include <random>
#include <string>
#include <memory>

#include "src/model/ffm/ffm_worker.h"

namespace xflow{
    void FFMWorker::calculate_pctr(int start,int end){
        size_t idx =0 ;
        size_t field_idx = 0;
        auto all_keys = std::vector<Base::sample_key>();
        auto unique_keys = std::vector<std::pair<ps::Key,ps::Key>>();
        int line_num = 0;
        auto all_fgids = std::vector<ps::Key>();
        auto all_fids = std::vector<ps::Key>();
        for(int row=start;row<end;row++){
            int sample_size = train_data->fea_matrix[row].size();
            Base::sample_key sk;
            sk.sid = line_num;
            for(int j=0;j<sample_size;j++){
                idx = train_data->fea_matrix[row][j].fid;
                field_idx = train_data->fea_matrix[row][j].fgid;
                sk.fid = idx;
                sk.fgid = field_idx;
                all_keys.push_back(sk);
                unique_keys.push_back({field_idx,idx});
            }
            ++line_num;
        }
        std::sort(all_keys.begin(),all_keys.end(),base_->field_sort_finder);
        std::sort((unique_keys).begin(),(unique_keys).end());
        (unique_keys).erase(unique((unique_keys).begin(),(unique_keys).end(),(unique_keys).end()));

        for(std::pair<ps::Key,ps::Key> a:unique_keys){
            all_fgids.push_back(a.first);
            all_fids.push_back(a.second);
        }

        int keys_size = (unique_keys).size();

        auto w = std::vector<float>();
        kv_w->Wait(kv_w->Pull(all_fids,&w));
        auto pair_keys = std::vector<ps::Key>();
        for(size_t i =0; i < all_fids.size();i++)
            pair_keys.push_back(i * f_dim_ + all_fgids[i]);
        auto v = std::vector<float>();
        kv_v->Wait(kv_v->Pull(pair_keys,&v));

        auto wx = std::vector<float>(line_num);
        for(int j=0,i=0;j<all_keys.size();){
            size_t all_keys_fid = all_keys[j].fid;
            size_t all_keys_fgid = all_keys[j].fgid;
            size_t weight_fid = all_fids[i];
            if(all_keys_fid == weight_fid){
                wx[all_keys[j].sid] += w[i];
                ++j;
            }else if(all_keys_fid > weight_fid){
                ++i;
            }
        }
        std::sort(all_keys.begin(),all_keys.end(),[](Base::sample_key& a,Base::sample_key& b){
            return a.sid < b.sid;
        });

        //建立all_keys和weight_fid的映射关系
        std::unordered_map<size_t,int> fid2id;
        for(int j=0,i=0;j<all_keys.size();j){
            size_t all_keys_fid = all_keys[j].fid;
            size_t all_keys_fgid = all_keys[j].fgid;
            size_t weight_fid = all_fids[i];
            if(all_keys_fid == weight_fid){
                fid2id[all_keys_fid] = i;
                ++j;
            }else if(all_keys_fid > weight_fid){
                ++i;
            }
        }


        for(int i=0,j=0;i<all_keys.size();){
            while(j<all_keys.size() && all_keys[i].sid == all_keys[j].sid) j++;
            j--;
            size_t sid = all_keys[i].sid;
            for(int m=i;m<=j;m++){
                for(int n =i+1;n<=j;n++){
                    for(int k=0;k<v_dim_;k++){
                        wx[sid] += v[( fid2id[all_keys[m].fid] * f_dim_ + all_keys[m].fgid) * v_dim_ + k] * v[( fid2id[all_keys[n].fid] * f_dim_+ all_keys[n].fgid) * v_dim_ + k];
                    }
                }
            }
            j++;
        }

        for(int i =0;i<wx.size();i++){
            float pctr = base_->sigmoid(wx[i]);
            Base::auc_key ak;
            ak.label = test_data->label[start++];
            ak.pctr = pctr;
            mutex.lock();
            test_auc_vec.push_back(ak);
            md << pctr << "\t" << 1 - ak.label << "\t" << ak.label << std::endl;
            mutex.unlock();
        }
        --calculate_pctr_thread_finish_num;
    }

    void FFMWorker::predict(ThreadPool* pool,int rank,int block){
        char buffer[1024];
        snprintf(buffer, 1024, "%d_%d", rank, block);
        std::string filename = buffer;
        md.open("pred_" + filename + ".txt");
        if (!md.is_open()) std::cout << "open pred file failure!" << std::endl;
        snprintf(test_data_path, 1024, "%s-%05d", test_file_path, rank);
        xflow::LoadData test_data_loader(test_data_path, ((size_t)2) << 20);
        test_data = &(test_data_loader.m_data);
        test_auc_vec.clear();
        while (true) {
            test_data_loader.load_minibatch_hash_data();
            if (test_data->fea_matrix.size() <= 0) break;
            int thread_size = test_data->fea_matrix.size() / core_num;
            calculate_pctr_thread_finish_num = core_num;
            for (int i = 0; i < core_num; ++i) {
            int start = i * thread_size;
            int end = (i + 1)* thread_size;
            pool->enqueue(std::bind(&FFMWorker::calculate_pctr, this, start, end));
            }
            while (calculate_pctr_thread_finish_num > 0) usleep(10);
        }
        md.close();
        test_data = NULL;
        base_->calculate_auc(test_auc_vec);
    }

    void FFMWorker::calculate_gradient(std::vector<Base::sample_key>& all_keys,std::vector<ps::Key>& all_fgids,std::vector<ps::Key>& all_fids,int start,int end,std::vector<float>& v,std::vector<float>& v_sum,std::vector<float>& loss,std::vector<float>& push_w_gradient,std::vector<float>& push_v_gradient){
        std::unordered_map<size_t,int> fid2id;
        for(int j=0,i=0;j<all_keys.size();){
            size_t all_keys_fid = all_keys[j].fid;
            size_t all_keys_fgid = all_keys[j].fgid;
            size_t weight_fid = all_fids[i];
            if(all_keys_fid == weight_fid){
                fid2id[all_keys_fid] = i;
                ++j;
            }else if(all_keys_fid > weight_fid){
                ++i;
            }
        }

        for (int j = 0, i = 0; j < all_keys.size(); ) {
            size_t allkeys_fid = all_keys[j].fid;
            size_t weight_fid = all_fids[i];
            int sid = all_keys[j].sid;
            if (allkeys_fid == weight_fid) {
                (push_w_gradient)[i] += loss[sid];
                ++j;
            } else if (allkeys_fid > weight_fid) {
                ++i;
            }
        }
        

        for(int i=0;i<all_keys.size();i++){
            size_t ffid = all_keys[i].fid;
            size_t ffgid = all_keys[i].fgid;
            size_t fsid = all_keys[i].sid;
            for(int j = i+1;j<all_keys.size();j++){
                size_t sfid = all_keys[j].fid;
                size_t sfgid = all_keys[j].fgid;
                size_t ssid = all_keys[j].sid;
                if(fsid != ssid) continue;
                for(int k=0;k<v_dim_;k++){
                    push_v_gradient[(i*f_dim_ + ffgid)*v_dim_ + k] += loss[fsid]* v[(j*f_dim_ + sfgid)*v_dim_ + k];
                }
            }
        }

        
        size_t line_num = end - start;
        for (size_t i = 0; i < (push_w_gradient).size(); ++i) {
            (push_w_gradient)[i] /= 1.0 * line_num;
        }
        for (size_t i = 0; i < push_v_gradient.size(); ++i) {
            push_v_gradient[i] /= 1.0 * line_num;
        }
        
    }

    void FFMWorker::calculate_loss(std::vector<float>& w,std::vector<float>& v,std::vector<Base::sample_key>& all_keys,std::vector<ps::Key>& all_fgids,std::vector<ps::Key>& all_fids,int start,int end,std::vector<float>& v_sum,std::vector<float>& loss){
        auto wx = std::vector<float>(end - start);
        for(int j=0,i=0;j<all_keys.size();){
            size_t all_keys_fid = all_keys[j].fid;
            size_t all_keys_fgid = all_keys[j].fgid;
            size_t weight_fid = all_fids[i];
            if(all_keys_fid == weight_fid){
                wx[all_keys[j].sid] += w[i];
                ++j;
            }else if(all_keys_fid > weight_fid){
                ++i;
            }
        }
        std::sort(all_keys.begin(),all_keys.end(),[](Base::sample_key& a,Base::sample_key& b){
            return a.sid < b.sid;
        });

        //建立all_keys和weight_fid的映射关系
        std::unordered_map<size_t,int> fid2id;
        for(int j=0,i=0;j<all_keys.size();j){
            size_t all_keys_fid = all_keys[j].fid;
            size_t all_keys_fgid = all_keys[j].fgid;
            size_t weight_fid = all_fids[i];
            if(all_keys_fid == weight_fid){
                fid2id[all_keys_fid] = i;
                ++j;
            }else if(all_keys_fid > weight_fid){
                ++i;
            }
        }


        for(int i=0,j=0;i<all_keys.size();){
            while(j<all_keys.size() && all_keys[i].sid == all_keys[j].sid) j++;
            j--;
            size_t sid = all_keys[i].sid;
            for(int m=i;m<=j;m++){
                for(int n =i+1;n<=j;n++){
                    for(int k=0;k<v_dim_;k++){
                        wx[sid] += v[( fid2id[all_keys[m].fid] * f_dim_ + all_keys[m].fgid) * v_dim_ + k] * v[( fid2id[all_keys[n].fid] * f_dim_+ all_keys[n].fgid) * v_dim_ + k];
                    }
                }
            }
            j++;
        }
        for(int i =0;i<wx.size();i++){
            float pctr = base_->sigmoid(wx[i]);
            loss[i] = pctr - train_data->label[start++];
        }
    }

    void FFMWorker::update(int start,int end){
        size_t idx =0 ;
        size_t field_idx = 0;
        auto all_keys = std::vector<Base::sample_key>();
        auto unique_keys = std::vector<std::pair<ps::Key,ps::Key>>();
        int line_num = 0;
        auto all_fgids = std::vector<ps::Key>();
        auto all_fids = std::vector<ps::Key>();
        for(int row=start;row<end;row++){
            int sample_size = train_data->fea_matrix[row].size();
            Base::sample_key sk;
            sk.sid = line_num;
            for(int j=0;j<sample_size;j++){
                idx = train_data->fea_matrix[row][j].fid;
                field_idx = train_data->fea_matrix[row][j].fgid;
                sk.fid = idx;
                sk.fgid = field_idx;
                all_keys.push_back(sk);
                unique_keys.push_back({field_idx,idx});
            }
            ++line_num;
        }
        std::sort(all_keys.begin(),all_keys.end(),base_->field_sort_finder);
        std::sort((unique_keys).begin(),(unique_keys).end());
        (unique_keys).erase(unique((unique_keys).begin(),(unique_keys).end(),(unique_keys).end()));

        for(std::pair<ps::Key,ps::Key> a:unique_keys){
            all_fgids.push_back(a.first);
            all_fids.push_back(a.second);
        }

        int keys_size = (unique_keys).size();

        auto w = std::vector<float>();
        kv_w->Wait(kv_w->Pull(all_fids,&w));
        auto push_w_gradient = std::vector<float>(keys_size);

        auto pair_keys = std::vector<ps::Key>();
        for(size_t i =0; i < all_fids.size();i++)
            pair_keys.push_back(i * f_dim_ + all_fgids[i]);
        auto v = std::vector<float>();
        kv_v->Wait(kv_v->Pull(pair_keys,&v));
        auto push_v_gradient = std::vector<float>(keys_size * f_dim_ * v_dim_);

        auto loss = std::vector<float>(end - start);
        auto v_sum = std::vector<float>(end - start);
        calculate_loss(w,v,all_keys,all_fgids,all_fids,start,end,v_sum,loss);
        calculate_gradient(all_keys, all_fgids, all_fids, start, end, v, v_sum, loss,
                     push_w_gradient, push_v_gradient);
        kv_w->Wait(kv_w->Push(all_fids,push_w_gradient));
        kv_v->Wait(kv_v->Push(pair_keys,push_v_gradient));

        --gradient_thread_finish_num;
    }

    void FFMWorker::batch_training(ThreadPool* pool){
        std::vector<ps::Key> key(1);
        std::vector<float> val_w(1);
        std::vector<float> val_v(f_dim_ * v_dim_);
        for(int epoch=0;epoch<epochs;epoch++){
            xflow::LoadData train_data_loader(train_data_path, block_size<<20);
            train_data = &(train_data_loader.m_data);
            int block = 0;
            while(1){
                train_data_loader.load_minibatch_hash_data(50);
                if(train_data->fea_matrix.size() <=0 ) break;
                int thread_size = train_data->fea_matrix.size() / core_num;
                gradient_thread_finish_num = core_num;
                for(int i =0;i<core_num;i++){
                    int start = i * thread_size;
                    int end = (i+1)*thread_size;
                    pool_->enqueue(std::bind(&FFMWorker::update,this,start,end));
                }
                while(gradient_thread_finish_num>0){
                    usleep(5);
                }
                ++block;
            }
            if ((epoch + 1) % 30 == 0) std::cout << "epoch : " << epoch << std::endl;
            train_data = NULL;
        }
    }

    void FFMWorker::train(){
        rank = ps::MyRank();
        std::cout<<"my rank is = "<< rank << std::endl;
        snprintf(train_data_path,1024,"%s-%05d", train_file_path, rank);
        batch_training(pool_);
        if(rank==0){
            std::cout<<"FFM AUC: "<<std::endl;
            predict(pool_,rank,0);
        }
        std::cout<<"train end......"<<std::endl;
    }
};