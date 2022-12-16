from apex.amp._amp_state import _amp_state
import contextlib
import torch

scaler = torch.cuda.amp.GradScaler()

@contextlib.contextmanager
def scale_loss(loss,
               optimizer,
               loss_id=0):

    loss_scaler = _amp_state.loss_scalers[loss_id]

    if not optimizer._amp_stash.params_have_scaled_gradients:
        optimizer._prepare_amp_backward()

    yield loss.float() * 4096

    loss_scaler.clear_overflow_state()

    optimizer._post_amp_backward(loss_scaler)
    optimizer._amp_stash.params_have_scaled_gradients = False

    # Probably ok to skip this if not delay_unscale
    if _amp_state.opt_properties.patch_torch_functions:
        _amp_state.handle._clear_cache()