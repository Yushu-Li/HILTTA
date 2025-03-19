import os
import logging
import numpy as np

from models.model import get_model
from hiltta import hiltta
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, get_domain_sequence, adaptation_method_lookup

from methods.tent import Tent
from methods.rmt import RMT
from methods.eata import EATA
from methods.sar import SAR
from methods.shot import SHOT
from methods.pl import PL

import torch
import torch.nn as nn
from models.model import split_up_model
from augmentations.transforms_cotta import get_tta_transforms

logger = logging.getLogger(__name__)


def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = [
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes)

    # setup test-time adaptation method
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    anchor_model = get_model(cfg, num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "imagenet_d109" and not cfg.CORRUPTION.TYPE[0]:
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    dom_names_loop = dom_names_all



    severities = cfg.CORRUPTION.SEVERITY

    errs = []
    domain_dict = {}
    active_pool = None
    # SGD optimizer
    

    lr_active=cfg.ACTIVE.LR
    if cfg.MODEL.ADAPTATION == "rmt":
        model.feature_extractor, model.classifier = split_up_model(model.model, cfg.MODEL.ARCH, model.dataset_name)
        model.feature_extractor_ema, model.classifier_ema = split_up_model(model.model_ema, cfg.MODEL.ARCH, model.dataset_name)
        model.active_optimizer = torch.optim.Adam(model.model_ema.parameters(),
                                    lr=lr_active,)
    else:
        model.feature_extractor, model.classifier = split_up_model(model.model, cfg.MODEL.ARCH, model.dataset_name)

    
        model.active_optimizer = torch.optim.Adam(model.model.parameters(),
                                        lr=lr_active,)
    
        
    
    model.select_transform = get_tta_transforms(cfg.CORRUPTION.DATASET, padding_mode='edge', cotta_augs=False, soft=True)

    model.select_queue = []
    
    
    # start evaluation
    for i_dom, domain_name in enumerate(dom_names_loop):
        if i_dom == 0:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(arch=cfg.MODEL.ARCH,
                                               setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               domain_names_all=dom_names_all,
                                               alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()),
                                               domain_id=i_dom,)

            # evaluate the model
            acc, domain_dict, model, active_pool = hiltta(cfg,
                                            model,
                                            data_loader=test_data_loader,
                                            domain_dict=domain_dict,
                                            anchor_model=anchor_model,
                                            active_pool=active_pool
                                            )

            err = 1. - acc
            errs.append(err)


            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}")


    logger.info(f"mean error: {np.mean(errs):.2%}")


if __name__ == '__main__':
    evaluate('"Evaluation.')

