

def prepare_criterion(criterion_params):
    from src.utils.mapping_new import CRITERION_NAME_MAP
    """
    Prepares the loss function (criterion) based on the provided parameters.

    Args:
        criterion_params (dict): 
            Parameters for the criterion, must contain:
                - 'criterion_name': str, key for CRITERION_NAME_MAP
                - 'criterion_params': dict, parameters for the criterion constructor

    Returns:
        torch.nn.Module: Instantiated criterion.
    """
    criterion_class = CRITERION_NAME_MAP[criterion_params['criterion_name']]
    criterion = criterion_class(**criterion_params['criterion_params'])
    return criterion