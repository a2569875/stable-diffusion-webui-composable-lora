from typing import List, Dict, Optional, Union
import re
import torch
import composable_lora_step
import composable_lycoris
import plot_helper
import lora_ext
from modules import extra_networks, devices

def lora_forward(compvis_module: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention], input, res):
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print
    global first_log_drawing
    global drawing_lora_first_index

    import lora

    if composable_lycoris.has_webui_lycoris:
        import lycoris
        if len(lycoris.loaded_lycos) > 0 and not first_log_drawing:
            print("Found LyCORIS models, Using Composable LyCORIS.")

    if not first_log_drawing:
        first_log_drawing = True
        if enabled:
            print("Composable LoRA load successful.")
        if opt_plot_lora_weight:
            log_lora()
            drawing_lora_first_index = drawing_data[0]

    if len(lora_ext.get_loaded_lora()) == 0:
        return res
    
    if hasattr(devices, "cond_cast_unet"):
        input = devices.cond_cast_unet(input)

    lora_layer_name_loading : Optional[str] = getattr(compvis_module, 'lora_layer_name', None)
    if lora_layer_name_loading is None:
        lora_layer_name_loading = getattr(compvis_module, 'network_layer_name', None)
    if lora_layer_name_loading is None:
        return res
    #let it type is actually a string
    lora_layer_name : str = str(lora_layer_name_loading)
    del lora_layer_name_loading

    lora_loaded_loras = lora_ext.get_loaded_lora()
    num_loras = len(lora_loaded_loras)
    if composable_lycoris.has_webui_lycoris:
        num_loras += len(lycoris.loaded_lycos)

    if text_model_encoder_counter == -1:
        text_model_encoder_counter = len(prompt_loras) * num_loras

    tmp_check_loras = [] #store which lora are already apply
    tmp_check_loras.clear()

    for m_lora in lora_loaded_loras:
        module = m_lora.modules.get(lora_layer_name, None)
        if module is None:
            #fix the lyCORIS issue
            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
            continue

        current_lora = composable_lycoris.normalize_lora_name(m_lora.name)
        lora_already_used = False
        if current_lora in tmp_check_loras:
            lora_already_used = True
        #store the applied lora into list
        tmp_check_loras.append(current_lora)
        if lora_already_used:
            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
            continue
        
        #support for lyCORIS
        patch = composable_lycoris.get_lora_patch(module, input, res, lora_layer_name)
        alpha = composable_lycoris.get_lora_alpha(module, 1.0)
        num_prompts = len(prompt_loras)

        # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
        res = apply_composable_lora(lora_layer_name, m_lora, module, "lora", patch, alpha, res, num_loras, num_prompts)
    return res

re_AND = re.compile(r"\bAND\b")

def load_prompt_loras(prompt: str):
    global is_single_block
    global full_controllers
    global first_log_drawing
    global full_prompt
    prompt_loras.clear()
    prompt_blocks.clear()
    lora_controllers.clear()
    drawing_data.clear()
    full_controllers.clear()
    drawing_lora_names.clear()
    cache_layer_list.clear()
    #load AND...AND block
    subprompts = re_AND.split(prompt)
    full_prompt = prompt
    tmp_prompt_loras = []
    tmp_prompt_blocks = []
    for i, subprompt in enumerate(subprompts):
        loras = {}
        _, extra_network_data = extra_networks.parse_prompt(subprompt)
        for m_type in ['lora', 'lyco']:
            if m_type in extra_network_data.keys():
                for params in extra_network_data[m_type]:
                    name = params.items[0]
                    multiplier = float(params.items[1]) if len(params.items) > 1 else 1.0
                    loras[f"{m_type}:{name}"] = multiplier

        tmp_prompt_loras.append(loras)
        tmp_prompt_blocks.append(subprompt)
    is_single_block = (len(tmp_prompt_loras) == 1)

    #load [A:B:N] syntax
    if opt_composable_with_step:
        print("Loading LoRA step controller...")
    tmp_lora_controllers = composable_lora_step.parse_step_rendering_syntax(prompt)

    #for batches > 1
    prompt_loras.extend(tmp_prompt_loras * num_batches)
    lora_controllers.extend(tmp_lora_controllers * num_batches)
    prompt_blocks.extend(tmp_prompt_blocks * num_batches)

    for controller_it in tmp_lora_controllers:
        full_controllers += controller_it
    first_log_drawing = False

def reset_counters():
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter
    global should_print

    # reset counter to uc head
    text_model_encoder_counter = -1
    diffusion_model_counter = 0
    step_counter += 1
    should_print = True
    
def reset_step_counters():
    global step_counter
    global should_print

    should_print = True
    step_counter = 0

def add_step_counters(): 
    global step_counter
    global should_print

    should_print = True
    step_counter += 1

    reset_flag = False
    if step_counter == num_steps + 1:
        if not opt_hires_step_as_global:
            step_counter = 0
            reset_flag = True
    elif step_counter > num_steps + num_hires_steps:
        step_counter = 0
        reset_flag = True
    if not reset_flag:
        if opt_plot_lora_weight:
            log_lora()

def log_lora():
    import lora
    loaded_loras = lora_ext.get_loaded_lora()
    loaded_lycos = []
    if composable_lycoris.has_webui_lycoris:
        import lycoris
        loaded_lycos = lycoris.loaded_lycos

    tmp_data : List[float] = []
    if len(loaded_loras) + len(loaded_lycos) <= 0:
        tmp_data = [0.0]
        if len(drawing_lora_names) <= 0:
            drawing_lora_names.append("LoRA Model Not Found.")
    for m_type in [("lora", loaded_loras), ("lyco", loaded_lycos)]:
        for m_lora in m_type[1]:
            m_lora_name = composable_lycoris.normalize_lora_name(m_lora.name)
            custom_scope = {}
            if opt_composable_with_step:
                custom_scope = {
                    "is_negative": False,
                    "lora": m_lora,
                    "lora_module": None,
                    "lora_type": m_type[0],
                    "lora_name": m_lora_name,
                    "lora_count": len(loaded_loras) + len(loaded_lycos),
                    "block_lora_count": len(loaded_loras) + len(loaded_lycos),
                    "layer_name": "ploting",
                    "current_prompt": full_prompt,
                    "sd_processing": sd_processing
                }
            current_lora = f"{m_type[0]}:{m_lora_name}"
            multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, "lora_layer_name")
            if opt_composable_with_step:
                multiplier = composable_lora_step.check_lora_weight(full_controllers, current_lora, step_counter, num_steps, custom_scope)
            index = -1
            if current_lora in drawing_lora_names:
                index = drawing_lora_names.index(current_lora)
            else:
                index = len(drawing_lora_names)
                drawing_lora_names.append(current_lora)
            if index >= len(tmp_data):
                for i in range(len(tmp_data), index):
                    tmp_data.append(0.0)
                tmp_data.append(multiplier)
            else:
                tmp_data[index] = multiplier
    drawing_data.append(tmp_data)

def plot_lora():
    """Plot the LoRA weight chart"""
    max_size = -1
    if len(drawing_data) < num_steps:
        item = drawing_data[len(drawing_data) - 1] if len(drawing_data) > 0 else [0.0]
        drawing_data.extend([item]*(num_steps - len(drawing_data)))
    drawing_data.insert(0, drawing_lora_first_index)
    for datalist in drawing_data:
        datalist_len = len(datalist)
        if datalist_len > max_size:
            max_size = datalist_len
    for i, datalist in enumerate(drawing_data):
        datalist_len = len(datalist)
        if datalist_len < max_size:
            drawing_data[i].extend([0.0]*(max_size - datalist_len))
    return plot_helper.plot_lora_weight(drawing_data, drawing_lora_names)

def lora_backup_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return
    import lora

    weights_backup = getattr(self, "composable_lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to(devices.cpu, copy=True), self.out_proj.weight.to(devices.cpu, copy=True))
        else:
            weights_backup = self.weight.to(devices.cpu, copy=True)

        self.composable_lora_weights_backup = weights_backup
        self.lora_weights_backup = weights_backup

def clear_cache_lora(compvis_module : Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention], force_clear : bool):
    lora_layer_name = getattr(compvis_module, 'lora_layer_name', 'unknown layer')
    if lora_layer_name in cache_layer_list:
        return
    cache_layer_list.append(lora_layer_name)
    lyco_weights_backup = getattr(compvis_module, "lyco_weights_backup", None)
    lora_weights_backup = getattr(compvis_module, "lora_weights_backup", None)
    composable_lora_weights_backup = getattr(compvis_module, "composable_lora_weights_backup", None)
    if enabled or force_clear:
        if composable_lora_weights_backup is not None:
            if isinstance(compvis_module, torch.nn.MultiheadAttention):
                compvis_module.in_proj_weight.copy_(composable_lora_weights_backup[0])
                compvis_module.out_proj.weight.copy_(composable_lora_weights_backup[1])
            else:
                compvis_module.weight.copy_(composable_lora_weights_backup)
        else:
            if lyco_weights_backup is not None:
                if isinstance(compvis_module, torch.nn.MultiheadAttention):
                    compvis_module.in_proj_weight.copy_(lyco_weights_backup[0])
                    compvis_module.out_proj.weight.copy_(lyco_weights_backup[1])
                    lora_weights_backup = (
                        lyco_weights_backup[0].to(devices.cpu, copy=True), 
                        lyco_weights_backup[1].to(devices.cpu, copy=True)
                    )
                else:
                    compvis_module.weight.copy_(lyco_weights_backup)
                    lora_weights_backup = lyco_weights_backup.to(devices.cpu, copy=True)
                setattr(compvis_module, "lora_weights_backup", lora_weights_backup)
            elif lora_weights_backup is not None:
                if isinstance(compvis_module, torch.nn.MultiheadAttention):
                    compvis_module.in_proj_weight.copy_(lora_weights_backup[0])
                    compvis_module.out_proj.weight.copy_(lora_weights_backup[1])
                else:
                    compvis_module.weight.copy_(lora_weights_backup)
            setattr(compvis_module, "lora_current_names", ())
            setattr(compvis_module, "lyco_current_names", ())
    else:
        if (composable_lora_weights_backup is not None) and composable_lycoris.has_webui_lycoris:
            if isinstance(compvis_module, torch.nn.MultiheadAttention):
                compvis_module.in_proj_weight.copy_(composable_lora_weights_backup[0])
                compvis_module.out_proj.weight.copy_(composable_lora_weights_backup[1])
            else:
                compvis_module.weight.copy_(composable_lora_weights_backup)

def apply_composable_lora(lora_layer_name, m_lora, module, m_type: str, patch, alpha, res, num_loras, num_prompts):
    global text_model_encoder_counter
    global diffusion_model_counter
    global step_counter

    custom_scope = {}
    if opt_composable_with_step:
        custom_scope = {
            "is_negative": False,
            "lora": m_lora,
            "lora_module": module,
            "lora_type": m_type,
            "lora_name": composable_lycoris.normalize_lora_name(m_lora.name),
            "lora_count": num_loras,
            "block_lora_count": 0,
            "layer_name": lora_layer_name,
            "current_prompt": "",
            "sd_processing": sd_processing
        }

    m_lora_name = f"{m_type}:{composable_lycoris.normalize_lora_name(m_lora.name)}"
    # print(f"lora.name={m_lora.name} lora.mul={m_lora.multiplier} alpha={alpha} pat.shape={patch.shape}")
    if enabled:
        if lora_layer_name.startswith("transformer_"):  # "transformer_text_model_encoder_"
            #
            if 0 <= text_model_encoder_counter // num_loras < len(prompt_loras):
                # c
                prompt_block_id = text_model_encoder_counter // num_loras
                loras = prompt_loras[prompt_block_id]
                multiplier = loras.get(m_lora_name, 0.0)
                if opt_composable_with_step:
                    custom_scope["current_prompt"] = prompt_blocks[prompt_block_id]
                    custom_scope["block_lora_count"] = len(loras)
                    lora_controller = lora_controllers[prompt_block_id]
                    multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora_name, -1, num_steps, custom_scope)
                if multiplier != 0.0:
                    multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                    # print(f"c #{text_model_encoder_counter // num_loras} lora.name={m_lora_name} mul={multiplier}  lora_layer_name={lora_layer_name}")
                    res = composable_lycoris.composable_forward(module, patch, alpha, multiplier, res)
            else:
                # uc
                multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                if (opt_uc_text_model_encoder or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                    # print(f"uc #{text_model_encoder_counter // num_loras} lora.name={m_lora_name} lora.mul={multiplier}  lora_layer_name={lora_layer_name}")
                    custom_scope["current_prompt"] = negative_prompt
                    custom_scope["is_negative"] = True
                    res = composable_lycoris.composable_forward(module, patch, alpha, multiplier, res)

            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)

        elif lora_layer_name.startswith("diffusion_model_"):  # "diffusion_model_"

            if res.shape[0] == num_batches * num_prompts + num_batches:
                # tensor.shape[1] == uncond.shape[1]
                tensor_off = 0
                uncond_off = num_batches * num_prompts
                for b in range(num_batches):
                    # c
                    for p, loras in enumerate(prompt_loras):
                        multiplier = loras.get(m_lora_name, 0.0)
                        if opt_composable_with_step:
                            prompt_block_id = p
                            custom_scope["current_prompt"] = prompt_blocks[prompt_block_id]
                            custom_scope["block_lora_count"] = len(loras)
                            lora_controller = lora_controllers[prompt_block_id]
                            multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora_name, step_counter, num_steps, custom_scope)
                        if multiplier != 0.0:
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                            # print(f"tensor #{b}.{p} lora.name={m_lora_name} mul={multiplier} lora_layer_name={lora_layer_name}")
                            res[tensor_off] = composable_lycoris.composable_forward(module, patch[tensor_off], alpha, multiplier, res[tensor_off])
                        tensor_off += 1

                    # uc
                    multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                    if (opt_uc_diffusion_model or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                        # print(f"uncond lora.name={m_lora_name} lora.mul={m_lora.multiplier} lora_layer_name={lora_layer_name}")
                        if is_single_block and opt_composable_with_step:
                            custom_scope["current_prompt"] = negative_prompt
                            custom_scope["is_negative"] = True
                            multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora_name, step_counter, num_steps, custom_scope)
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                        res[uncond_off] = composable_lycoris.composable_forward(module, patch[uncond_off], alpha, multiplier, res[uncond_off])
                    
                    uncond_off += 1
            else:
                # tensor.shape[1] != uncond.shape[1]
                cur_num_prompts = res.shape[0]
                base = (diffusion_model_counter // cur_num_prompts) // num_loras * cur_num_prompts
                prompt_len = len(prompt_loras)
                if 0 <= base < len(prompt_loras):
                    # c
                    for off in range(cur_num_prompts):
                        if base + off < prompt_len:
                            loras = prompt_loras[base + off]
                            multiplier = loras.get(m_lora_name, 0.0)
                            if opt_composable_with_step:
                                prompt_block_id = base + off
                                custom_scope["current_prompt"] = prompt_blocks[prompt_block_id]
                                custom_scope["block_lora_count"] = len(loras)
                                lora_controller = lora_controllers[prompt_block_id]
                                multiplier = composable_lora_step.check_lora_weight(lora_controller, m_lora_name, step_counter, num_steps, custom_scope)
                            if multiplier != 0.0:
                                multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                                # print(f"c #{base + off} lora.name={m_lora_name} mul={multiplier} lora_layer_name={lora_layer_name}")
                                res[off] = composable_lycoris.composable_forward(module, patch[off], alpha, multiplier, res[off])
                else:
                    # uc
                    multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
                    if (opt_uc_diffusion_model or (is_single_block and (not opt_single_no_uc))) and multiplier != 0.0:
                        # print(f"uc {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
                        if is_single_block and opt_composable_with_step:
                            custom_scope["current_prompt"] = negative_prompt
                            custom_scope["is_negative"] = True
                            multiplier = composable_lora_step.check_lora_weight(full_controllers, m_lora_name, step_counter, num_steps, custom_scope)
                            multiplier *= composable_lycoris.lycoris_get_multiplier_normalized(m_lora, lora_layer_name)
                        res = composable_lycoris.composable_forward(module, patch, alpha, multiplier, res)

            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
        else:
            # default
            multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
            if multiplier != 0.0:
                # print(f"default {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
                res = composable_lycoris.composable_forward(module, patch, alpha, multiplier, res)
            composable_lycoris.check_lycoris_end_layer(lora_layer_name, res, num_loras)
    else:
        # default
        multiplier = composable_lycoris.lycoris_get_multiplier(m_lora, lora_layer_name)
        if multiplier != 0.0:
            # print(f"DEFAULT {lora_layer_name} lora.name={m_lora_name} lora.mul={m_lora.multiplier}")
            res = composable_lycoris.composable_forward(module, patch, alpha, multiplier, res)
    return res

def lora_Linear_forward(self, input):
    if composable_lycoris.has_webui_lycoris:
        lora_backup_weights(self)
        if not enabled:
            import lycoris
            import lora
            lyco_count = len(lycoris.loaded_lycos)
            old_lyco_count = getattr(self, "old_lyco_count", 0)
            if old_lyco_count > 0 and lyco_count <= 0:
                clear_cache_lora(self, True)
            self.old_lyco_count = lyco_count
            lora_ext.load_lora_ext()
            torch.nn.Linear_forward_before_lyco = lora_ext.lora_Linear_forward
            torch.nn.Linear_forward_before_network = Linear_forward_before_clora
            #if lyco_count <= 0:
            #    return lora_ext.lora_Linear_forward(self, input)
            if 'lyco_notfound' in locals() or 'lyco_notfound' in globals():
                if lyco_notfound:
                    backup_Linear_forward = torch.nn.Linear_forward_before_lora
                    torch.nn.Linear_forward_before_lora = Linear_forward_before_clora
                    result = lycoris.lyco_Linear_forward(self, input)
                    torch.nn.Linear_forward_before_lora = backup_Linear_forward
                    return result
            return lycoris.lyco_Linear_forward(self, input)
    if lora_ext.is_sd_1_5:
        import networks
        networks.network_restore_weights_from_backup(self)
        networks.network_reset_cached_weight(self)
    else:
        clear_cache_lora(self, False)
    
    if (not self.weight.is_cuda) and input.is_cuda: #if variables not on the same device (between cpu and gpu)
        self_weight_cuda = self.weight.to(device=devices.device) #pass to GPU
        to_del = self.weight
        self.weight = None                    #delete CPU variable
        del to_del
        del self.weight                       #avoid pytorch 2.0 throwing exception
        self.weight = self_weight_cuda        #load GPU data to self.weight

    if devices.device.type == 'mps': #ensure input and weights both live on mps in the same format
        input = input.to(device=devices.device, dtype=devices.dtype)
        self_weight_cuda = self.weight.to(device=devices.device, dtype=devices.dtype) #pass to MPS
        to_del = self.weight
        self.weight = None                    #delete CPU variable
        del to_del
        del self.weight                       #avoid pytorch 2.0 throwing exception
        self.weight = self_weight_cuda        #load GPU data to self.weight
    
    res = torch.nn.Linear_forward_before_lora(self, input)
    res = lora_forward(self, input, res)
    if composable_lycoris.has_webui_lycoris:
        res = composable_lycoris.lycoris_forward(self, input, res)
    return res

def lora_Conv2d_forward(self, input):
    if composable_lycoris.has_webui_lycoris:
        lora_backup_weights(self)
        if not enabled:
            import lycoris
            import lora
            lyco_count = len(lycoris.loaded_lycos)
            old_lyco_count = getattr(self, "old_lyco_count", 0)
            if old_lyco_count > 0 and lyco_count <= 0:
                clear_cache_lora(self, True)
            self.old_lyco_count = lyco_count
            lora_ext.load_lora_ext()
            torch.nn.Conv2d_forward_before_lyco = lora_ext.lora_Conv2d_forward
            torch.nn.Conv2d_forward_before_network = Conv2d_forward_before_clora
            #if lyco_count <= 0:
            #    return lora_ext.lora_Conv2d_forward(self, input)
            if 'lyco_notfound' in locals() or 'lyco_notfound' in globals():
                if lyco_notfound:
                    backup_Conv2d_forward = torch.nn.Conv2d_forward_before_lora
                    torch.nn.Conv2d_forward_before_lora = Conv2d_forward_before_clora
                    result = lycoris.lyco_Conv2d_forward(self, input)
                    torch.nn.Conv2d_forward_before_lora = backup_Conv2d_forward
                    return result

            return lycoris.lyco_Conv2d_forward(self, input)
    if lora_ext.is_sd_1_5:
        import networks
        networks.network_restore_weights_from_backup(self)
        networks.network_reset_cached_weight(self)
    else:
        clear_cache_lora(self, False)
    if (not self.weight.is_cuda) and input.is_cuda:
        self_weight_cuda = self.weight.to(device=devices.device)
        to_del = self.weight
        self.weight = None
        del to_del
        del self.weight #avoid "cannot assign XXX as parameter YYY (torch.nn.Parameter or None expected)"
        self.weight = self_weight_cuda

    if devices.device.type == 'mps': #ensure input and weights both live on mps in the same format
        input = input.to(device=devices.device, dtype=devices.dtype)
        self_weight_cuda = self.weight.to(device=devices.device, dtype=devices.dtype) #pass to MPS
        to_del = self.weight
        self.weight = None                    #delete CPU variable
        del to_del
        del self.weight                       #avoid pytorch 2.0 throwing exception
        self.weight = self_weight_cuda        #load GPU data to self.weight

    res = torch.nn.Conv2d_forward_before_lora(self, input)
    res = lora_forward(self, input, res)
    if composable_lycoris.has_webui_lycoris:
        res = composable_lycoris.lycoris_forward(self, input, res)
    return res

def lora_MultiheadAttention_forward(self, input):
    if composable_lycoris.has_webui_lycoris:
        lora_backup_weights(self)
        if not enabled:
            import lycoris
            import lora
            lyco_count = len(lycoris.loaded_lycos)
            old_lyco_count = getattr(self, "old_lyco_count", 0)
            if old_lyco_count > 0 and lyco_count <= 0:
                clear_cache_lora(self, True)
            self.old_lyco_count = lyco_count
            lora_ext.load_lora_ext()
            torch.nn.MultiheadAttention_forward_before_lyco = lora_ext.lora_MultiheadAttention_forward
            torch.nn.MultiheadAttention_forward_before_network = MultiheadAttention_forward_before_clora

            #if lyco_count <= 0:
            #    return lora_ext.lora_MultiheadAttention_forward(self, input)
            if 'lyco_notfound' in locals() or 'lyco_notfound' in globals():
                if lyco_notfound:
                    backup_MultiheadAttention_forward = torch.nn.MultiheadAttention_forward_before_lora
                    torch.nn.MultiheadAttention_forward_before_lora = MultiheadAttention_forward_before_clora
                    result = lycoris.lyco_MultiheadAttention_forward(self, input)
                    torch.nn.MultiheadAttention_forward_before_lora = backup_MultiheadAttention_forward
                    return result

            return lycoris.lyco_MultiheadAttention_forward(self, input)
    if lora_ext.is_sd_1_5:
        import networks
        networks.network_restore_weights_from_backup(self)
        networks.network_reset_cached_weight(self)
    else:
        clear_cache_lora(self, False)
    if (not self.weight.is_cuda) and input.is_cuda:
        self_weight_cuda = self.weight.to(device=devices.device)
        to_del = self.weight
        self.weight = None
        del to_del
        del self.weight #avoid "cannot assign XXX as parameter YYY (torch.nn.Parameter or None expected)"
        self.weight = self_weight_cuda
    res = torch.nn.MultiheadAttention_forward_before_lora(self, input)
    res = lora_forward(self, input, res)
    if composable_lycoris.has_webui_lycoris:
        res = composable_lycoris.lycoris_forward(self, input, res)
    return res

def noop():
    pass

def should_reload():
    #pytorch 2.0 should reload
    match = re.search(r"\d+(\.\d+)?",str(torch.__version__)) 
    if not match:
        return True
    ver = float(match.group(0))
    return ver >= 2.0

enabled : bool = False
opt_composable_with_step : bool = False
opt_uc_text_model_encoder : bool = False
opt_uc_diffusion_model : bool = False
opt_plot_lora_weight : bool = False
opt_single_no_uc : bool = False
opt_hires_step_as_global : bool = False
verbose : bool = True

sd_processing = None
full_prompt: str = ""
negative_prompt: str = ""
drawing_lora_names : List[str] = []
drawing_data : List[List[float]] = []
drawing_lora_first_index : List[float] = []
first_log_drawing : bool = False

is_single_block : bool = False
num_batches: int = 0
num_steps: int = 20
num_hires_steps: int = 20
prompt_loras: List[Dict[str, float]] = []
text_model_encoder_counter: int = -1
diffusion_model_counter: int = 0
step_counter: int = 0
cache_layer_list : List[str] = []

should_print : bool = True
prompt_blocks: List[str] = []
lora_controllers: List[List[composable_lora_step.LoRA_Controller_Base]] = []
full_controllers: List[composable_lora_step.LoRA_Controller_Base] = []
