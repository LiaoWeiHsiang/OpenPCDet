from .detector3d_template import Detector3DTemplate
import numpy as np
import json

class PVRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

        self.model_for_cam = self.module_list[:7]
        # print(self.model_for_cam)
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # cls_preds_for_heatmap = batch_dict['cls_preds_for_heatmap']
        # print(cls_preds_for_heatmap)
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts,batch_dict

    def get_training_loss(self):
        # disp_dict = {}
        # loss_rpn, tb_dict,cls_loss,box_loss = self.dense_head.get_loss()
        # loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        # loss_rcnn, tb_dicti,rcnn_loss_cls,rcnn_loss_reg = self.roi_head.get_loss(tb_dict)
        
     
        
        # loss = loss_rpn + loss_point + loss_rcnn
        # loss_dict = {'loss_rpn':float(loss_rpn),'loss_point':float(loss_point),'loss_rcnn':float(loss_rcnn),'total_loss':float(loss),'cls_loss':float(cls_loss),'box_loss':float(box_loss),'rcnn_loss_cls':float(rcnn_loss_cls),'rcnn_loss_reg':float(rcnn_loss_reg)}
        # with open("/home/950154_customer/david/OpenPCDet/output/loss.json", "a") as outfile:  
            # np.savetxt(f, loss_array)
            # json.dump(loss_dict,outfile)
            # outfile.write("\n")        
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_point, tb_dict = self.point_head.get_loss(tb_dict)
        loss_rcnn, tb_dicti,rcnn_loss_cls,rcnn_loss_reg = self.roi_head.get_loss(tb_dict)
        # loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        # gnd_loss = self.dense_head.get_ground_loss_v2(batch_dict["voxel_coords"])
        # print("gnd_loss = {}".format(gnd_loss))
        print("loss_rpn = {}".format(loss_rpn))
        print("loss_point = {}".format(loss_point))
        print("loss_rcnn = {}".format(loss_rcnn))
        print("==============")
        # print(loss_rpn)
        loss = loss_rpn + loss_point + loss_rcnn  # + gnd_loss
        # loss = gnd_loss
        return loss, tb_dict, disp_dict
        
        # return loss, tb_dict, disp_dict
