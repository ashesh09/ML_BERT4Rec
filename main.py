# import torch

# from options import args
# from models import model_factory
# from dataloaders import dataloader_factory
# from trainers import trainer_factory
# from utils import *


# def train():
#     export_root = setup_train(args)
#     train_loader, val_loader, test_loader = dataloader_factory(args)
#     model = model_factory(args)
#     trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
#     trainer.train()

#     test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
#     if test_model:
#         trainer.test()


# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     else:
#         raise ValueError('Invalid mode')

# import torch
# from options import args
# from models import model_factory
# from dataloaders import dataloader_factory
# from trainers import trainer_factory
# from datasets import dataset_factory
# from utils import setup_train

# def train():
#     # Set up the experiment export folder, logging, etc.
#     export_root = setup_train(args)
    
#     # Load the dataset to extract genre mapping information.
#     # (Assumes that your dataset preprocessing stores the field 'genre_mapping'.)
#     dataset = dataset_factory(args).load_dataset()
    
#     # Extract the set of genre IDs from the mapping.
#     genre_ids = set(dataset['genre_mapping'].values())
    
#     # If your genre IDs start at 1 (and you reserve 0 for padding), then add one.
#     # Otherwise, if they’re zero-indexed, you might simply do: len(genre_ids)
#     args.genre_vocab_size = max(genre_ids) + 1
#     print("Genre vocab size set to:", args.genre_vocab_size)
    
#     # Create the dataloaders, model, and trainer.
#     train_loader, val_loader, test_loader = dataloader_factory(args)
#     model = model_factory(args)
#     trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    
#     # Start training.
#     trainer.train()
    
#     # Optionally test the model on the test set.
#     test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
#     if test_model:
#         trainer.test()

# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     else:
#         raise ValueError('Invalid mode')



# import torch
# from options import args
# from models import model_factory
# from dataloaders import dataloader_factory
# from trainers import trainer_factory
# from utils import setup_train
# from datasets import dataset_factory

# def train():
#     export_root = setup_train(args)
    
#     # Create the dataset object and force preprocessing (if necessary, delete old preprocessed file)
#     dataset_obj = dataset_factory(args)
#     data_dict = dataset_obj.load_dataset()
    
#     # Now update args with the computed genre vocabulary size
#     args.genre_vocab_size = data_dict['genre_vocab_size']
    
#     train_loader, val_loader, test_loader = dataloader_factory(args)
#     model = model_factory(args)
#     trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    
#     trainer.train()
    
#     test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
#     if test_model:
#         trainer.test()

# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     else:
#         raise ValueError('Invalid mode')


# import torch
# from options import args
# from models import model_factory
# from dataloaders import dataloader_factory
# from trainers import trainer_factory
# from utils import setup_train
# from datasets import dataset_factory

# def train():
#     export_root = setup_train(args)
    
#     # Create the dataset object and load the dictionary
#     dataset_obj = dataset_factory(args)
#     data_dict = dataset_obj.load_dataset()
#     args.genre_vocab_size = data_dict['genre_vocab_size']  # Update the args with the computed genre vocab size
    
#     # Now build the data loaders and model
#     train_loader, val_loader, test_loader = dataloader_factory(args)
#     model = model_factory(args)
#     trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    
#     trainer.train()
    
#     test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
#     if test_model:
#         trainer.test()

# if __name__ == '__main__':
#     if args.mode == 'train':
#         train()
#     else:
#         raise ValueError('Invalid mode')



import torch
from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from datasets import dataset_factory
from utils import setup_train

def train():
    export_root = setup_train(args)
    # Load the dataset so we can update args with genre_vocab_size.
    dataset_obj = dataset_factory(args)
    data_dict = dataset_obj.load_dataset()  # This returns a dictionary.
    if 'genre_vocab_size' in data_dict:
        args.genre_vocab_size = data_dict['genre_vocab_size']
    else:
        print("Warning: 'genre_vocab_size' not found in dataset; using default value.")
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    test_model = (input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')




