newest_tensorboard=`cd runs && ls -t | head -n1`
echo "run $newest_tensorboard"
tensorboard --logdir=runs/$newest_tensorboard