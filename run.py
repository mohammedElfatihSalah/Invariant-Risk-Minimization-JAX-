import jax
import haiku as hk
import optax
import functools
from dataset import data_stream, build_env
from baseline import build_forward_fn, GradientUpdater, loss_fn, accuracy, evaluate
import argparse


def main(args):
    # Create the dataset.
    print("...building the env...")
    env = build_env()
    print("...Finished building the env :)...")

    # Set up the model, loss, and updater.
    train_dataset = data_stream(env.train_data[0], 256)

    forward_fn = build_forward_fn(
      hidden_dim =args.hidden_dim, 
      nb_layers=args.num_layers
      )
    forward_fn = hk.transform(forward_fn)
    lm_loss_fn = functools.partial(loss_fn, forward_fn.apply)
    optimizer = optax.adam(args.lr)
    updater = GradientUpdater(forward_fn.init, lm_loss_fn, optimizer)
    
    # Initialize parameters.
    rng = jax.random.PRNGKey(10)
    data = next(train_dataset)
    state = updater.init(rng, data)
    for step in range(args.train_steps):
      w = args.weight_penalty if step > 190 else 1.0
      data1 = env.train_data[0]
      data2 = env.train_data[1]
      state, metrics = updater.update(state, data1, data2,w)  
      params = state['params']
    
      if step % args.evaluate_every == 0:
        train_avg_accuracy = 0
        nb_train_env = 0
        for data in env.train_data:
          train_avg_accuracy += evaluate(state['params'], rng, forward_fn, data)
          nb_train_env +=1
        train_avg_accuracy = train_avg_accuracy / nb_train_env

        test_avg_accuracy = 0
        nb_test_env = 0
        for data in env.test_data:
          test_avg_accuracy += evaluate(state['params'], rng, forward_fn, data)
          nb_test_env +=1
        test_avg_accuracy = test_avg_accuracy / nb_test_env
        print(f"step: {step+1} | train accuracy: {train_avg_accuracy * 100} | test accuracy: {test_avg_accuracy}")
      


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--name', type=str, default='ERM')
  parser.add_argument('--hidden_dim', type=int, default=390)
  parser.add_argument('--lr', type=float, default=.0002)
  parser.add_argument('--l2_loss', type=float, default=.0008)
  parser.add_argument('--weight_penalty', type=float, default=1e4)
  parser.add_argument('--train_steps', type=int, default=5001)
  parser.add_argument('--evaluate_every', type=int, default=100)
  parser.add_argument('--num_layers', type=int, default=3)

  args = parser.parse_args()
  print('='*35)
  print(f"Running Experiment {args.name}")
  print('='*35)
  main(args)
      