import torch
import torch.nn as nn
import numpy as np
import copy
import logging


logger = logging.getLogger(__name__)




def k_margin(final_output, feature, active_rate):
    def uncertainty_margin_confidence(output):
        """ 
        Returns the uncertainty score of a probability distribution using
        margin confidence 
                    
        """
        
        return 1 - (output.softmax(1).max(1)[0] - output.softmax(1).topk(2, 1)[0][:, 1])



    def KCenterGreedy_mat(embeddings, margin, n):
        """ 
        Returns the query index of a probability distribution using
        kcenter_greedy 
                    
        """
        embeddings = embeddings.detach().cpu() * torch.Tensor(margin[:, np.newaxis])

        embeddings = embeddings.detach().cpu().numpy()
        labeled_idxs = np.zeros(len(embeddings)+1, dtype=bool)
        embeddings = np.append(np.random.rand(1,len(embeddings[0]))*0, embeddings, 0)
        labeled_idxs[0] = 1


        dist_mat = np.linalg.norm(embeddings[:, np.newaxis] - embeddings, axis=2)

        mat = dist_mat[~labeled_idxs, :][:, labeled_idxs]


        for i in range(n):
            mat_min = mat.min(axis=1)
            q_idx_ = mat_min.argmax()
            q_idx = np.arange(len(labeled_idxs))[~labeled_idxs][q_idx_]
            labeled_idxs[q_idx] = True
            mat = np.delete(mat, q_idx_, 0)
            mat = np.append(mat, dist_mat[~labeled_idxs, q_idx][:, None], axis=1)



        return np.arange(len(embeddings)-1)[labeled_idxs[1:]]




    scores = uncertainty_margin_confidence(final_output).detach().cpu().numpy()

    scores_norm = scores 



    labeled_index = torch.tensor(KCenterGreedy_mat(feature, scores_norm, n=int(final_output.shape[0] * active_rate)))

    unlabeled_index = torch.tensor([i for i in range(final_output.shape[0]) if i not in labeled_index])

    

    return labeled_index.cpu(), unlabeled_index.cpu()


def hiltta(cfg,
                model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                domain_dict: dict,
                device: torch.device = None,
                anchor_model = None,
                active_pool = None,
                ):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_selection = cfg.ACTIVE.MODEL_SELECTION
    model_selection_choice = cfg.ACTIVE.MODEL_SELECTION_CHOICE
    active_rate = cfg.ACTIVE.RATE # active learning rate

    correct = 0.

    with torch.no_grad():
        for i, data in enumerate(data_loader):


            saved_model = []

            imgs, labels = data[0], data[1]
            

            if cfg.MODEL.ADAPTATION == 'rmt' or cfg.MODEL.ADAPTATION == 'sar':
                saved_model.append((model.copy_model_and_optimizer()))
            else:
                saved_model.append(copy.deepcopy(model))

            if cfg.MODEL.ADAPTATION == 'rmt':
                final_output = (model.model(imgs.to(device)) + model.model_ema(imgs.to(device)))
                feature = model.feature_extractor_ema(imgs.to(device)) 
            else:
                feature = model.feature_extractor(imgs.to(device))
                final_output = model.classifier(feature)


            labeled_index, unlabeled_index = k_margin(final_output, feature, active_rate)
           
            active_pool = [imgs[labeled_index], labels[labeled_index]]
            for choice in model_selection_choice:

                # change hyperparameters
                if model_selection == 'LR':
                    new_lr = float(choice)
                    for param_group in model.optimizer.param_groups:
                        param_group['lr'] = new_lr

                elif model_selection == 'D_MARGIN':
                    model.d_margin = float(choice)

                elif model_selection == 'LAMBDA_CONT':
                    model.lambda_cont = float(choice)

                elif model_selection == 'THETA':
                    model.theta = float(choice)

                elif model_selection == 'THRESHOLD':
                    model.threshold = float(choice)
                

                
                output = model([imgs.to(device)[unlabeled_index],labels.long().to(device)[unlabeled_index], 
                                imgs.to(device)[labeled_index],labels.long().to(device)[labeled_index]])



                if cfg.MODEL.ADAPTATION == 'rmt' or cfg.MODEL.ADAPTATION == 'sar':
                    saved_model.append((model.copy_model_and_optimizer()))
                else:

                    saved_model.append(copy.deepcopy(model))

            img_test = imgs.to(device)
            labels = labels.long().to(device)




            best_model_index, select_queue = active_model_selection(cfg, model, img_test, saved_model, anchor_model, labeled_index, unlabeled_index, active_pool=active_pool, score_queue= model.select_queue)
            
            if cfg.MODEL.ADAPTATION == 'rmt' or cfg.MODEL.ADAPTATION == 'sar':
                for modele, model_state in zip(model.models, saved_model[best_model_index][0]):
                    modele.load_state_dict(model_state, strict=True)
                model.optimizer.load_state_dict(saved_model[best_model_index][1])
            else:

                model = copy.deepcopy(saved_model[best_model_index])

            model.select_queue = select_queue

            supervised_training(model, img_test, labels, labeled_index, method=cfg.MODEL.ADAPTATION)

            

            predictions = final_output.argmax(1)

            correct += (predictions == labels.to(device)).float().sum()


    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict, model, active_pool



def active_model_selection(cfg, model, imgs_test, saved_model, anchor_model, labeled_index, unlabeled_index, active_pool=[], score_queue=[]):

    
    if len(labeled_index) == 0:
        return 1, score_queue
    transform = model.select_transform



    

    with torch.no_grad():

        begin = 1
        best_model = 1      
                

        for i in range(begin, len(saved_model)):
        
        
            if cfg.MODEL.ADAPTATION == 'rmt'  or cfg.MODEL.ADAPTATION == 'sar':
                for modele, model_state in zip(model.models, saved_model[i][0]):
                        modele.load_state_dict(model_state, strict=True)

            else:
                model = copy.deepcopy(saved_model[i])
            
           
            input_img = torch.cat([imgs_test[unlabeled_index], active_pool[0].to(imgs_test.device)], 0)

            if cfg.MODEL.ADAPTATION == "rmt":
                outputs = model.model(input_img)
            else:
                outputs = model.model(input_img)
            anchor_outputs = anchor_model(input_img)


            anchor_loss = nn.MSELoss()(torch.softmax(outputs[:len(unlabeled_index)],1), torch.softmax(anchor_outputs[:len(unlabeled_index)],1))

            ce_loss = nn.CrossEntropyLoss()(outputs[len(unlabeled_index):], active_pool[1].long().to(imgs_test.device)).item()

            if i == begin:
                queue_1 = [anchor_loss.item()]
                queue_2 = [ce_loss]
            else:
                queue_1.append(anchor_loss.item())
                queue_2.append(ce_loss)

        anchor_loss_rank = np.array(queue_1)
        anchor_loss_rank = anchor_loss_rank - anchor_loss_rank.min()

        if anchor_loss_rank.max() == 0:
            anchor_loss_rank = anchor_loss_rank + 1e-6
        anchor_loss_rank = anchor_loss_rank / (anchor_loss_rank.max())

            
        prob_rank = np.array(queue_2)

        prob_rank = prob_rank - prob_rank.min()

        if prob_rank.max() == 0:

            prob_rank = prob_rank + 1e-6
        prob_rank = prob_rank / (prob_rank.max())
        
        total_rank = np.array(anchor_loss_rank) + np.array(prob_rank)



        if len(score_queue) ==0:
            score_queue.append(total_rank)
        else:
            # EMA update
            for j in range(len(score_queue[0])):
                score_queue[0][j] = score_queue[0][j] * float(cfg.ACTIVE.MODEL_SELECTION_MOMENTUM) + total_rank[j] * (1 - float(cfg.ACTIVE.MODEL_SELECTION_MOMENTUM))

        total_rank = np.array(score_queue).mean(0)


        
        
        

        
        
        best_model = np.argmin(total_rank) + begin

        

        
               

            
            
    return best_model, score_queue

@torch.enable_grad()
def supervised_training(model, img_input, label, labeled_mask, method):

    if method == 'rmt' :
        if len(labeled_mask) != 0:
            model.model_ema.requires_grad_(True)
            outputs = model.model_ema(img_input)
            loss = nn.CrossEntropyLoss()(outputs[labeled_mask], label[labeled_mask])
            loss.backward()
            model.active_optimizer.step()
            model.active_optimizer.zero_grad()
            model.model_ema.requires_grad_(False)


    else:
        if len(labeled_mask) != 0:
            model.model.requires_grad_(True)
            outputs = model.model(img_input)

            
            loss = nn.CrossEntropyLoss()(outputs[labeled_mask], label[labeled_mask])
            loss.backward()
            model.active_optimizer.step()
            model.active_optimizer.zero_grad()
            model.configure_model()