class Config:
    
    # dataset
    dataset_name = "BSD68"
    image_size = 128
    
    # noise settings (degradation model)
    noise_sigma = 25
    
    # training parameters
    batch_size = 64
    epochs = 5
    learning_rate = 0.0002
    
    # optimizer
    optimizer = "Adam"
    
    # reproducibility
    random_seed = 42
    
    # device
    device = "cuda"
