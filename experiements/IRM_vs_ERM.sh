
python -m _src.run \
  --name='IRM'\
  --train_steps=6001\
  --hidden_dim=390\
  --num_layers=3\
  --lr=.0001\
  --l2_loss=.0002\
  --weight_penalty=1e4\

echo "============  Finished ====================="

python -m _src.run \
  --name='ERM'\
  --train_steps=6001\
  --hidden_dim=390\
  --num_layers=3\
  --lr=.0001\
  --l2_loss=.0002\
  --weight_penalty=1\

echo "============  Finished ====================="
