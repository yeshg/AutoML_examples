# As of 10/12/2019: One caveat of using TF2.0 is that TF AutoGraph
# functionality does not interact nicely with Ray actors. One way to get around
# this is to `import tensorflow` inside the Tune Trainable.
#
import os, json, argparse, random, time
import multiprocessing
import numpy as np

from filelock import FileLock

import ray
import ray.tune as tune
from ray.tune import Trainable
from ray.tune.utils import validate_save_restore
from ray.tune.schedulers import HyperBandScheduler, ASHAScheduler

TRIALS = 5

class MyTrainableClass(Trainable):
    """Optimize inference time of resnet50
    """

    def _setup(self, config):
        # IMPORTANT: See the above note.
        import tensorflow as tf

        # config environment variables, tensorflow runtime settings
        os.environ["OMP_NUM_THREADS"] = str(config["OMP_NUM_THREADS"])
        # os.environ["KMP_BLOCKTIME"] = str(config["KMP_BLOCKTIME"])
        # os.environ["KMP_AFFINITY"] = str(config["KMP_AFFINITY"])
        self.inter_op_parallelism_threads = config["inter_op_parallelism_threads"]
        self.intra_op_parallelism_threads = config["intra_op_parallelism_threads"]
        tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism_threads)
        tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism_threads)

        # other config
        self.warmup = config["warmup"]
        self.batch_size = config["batch_size"]

        def get_resnet_model():
            if not os.path.exists("/tmp/model"):
                with FileLock(os.path.expanduser("/tmp/model.lock")):
                    model = tf.keras.applications.ResNet50(weights='imagenet')
                    model.save('/tmp/model')
                    print("Done fetching model")
            loaded_model = tf.keras.models.load_model('/tmp/model')
            return loaded_model

        # prepare model
        self.model = get_resnet_model()

        @tf.function
        def test_step(data):
            predictions = self.model(data)
        self.tf_test_step = test_step

        # prepare test data
        self.data = np.random.random((self.batch_size, 224, 224, 3))

        # track tune progress
        self.timestep = 0

    def _train(self):
        
        # run model and calculate time elapsed
        if self.warmup:
            _ = self.tf_test_step(self.data)

        avg_t = 0
        for _ in range(TRIALS):
            tm = time.time()
            _ = self.tf_test_step(self.data)
            tm = time.time() - tm
            avg_t += tm
        avg_t /= TRIALS

        self.timestep += 1

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        return {"latency": avg_t, "throughput": self.batch_size/avg_t}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_args")
        with open(checkpoint_path, "w") as f:
            f.write(json.dumps({
                "timestep": self.timestep,
                "batch_size": self.batch_size,
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "inter_op_parallelism_threads": self.inter_op_parallelism_threads,
                "intra_op_parallelism_threads": self.intra_op_parallelism_threads}))
        return checkpoint_dir

    def _restore(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_args")
        with open(checkpoint_path) as f:
            foo = json.loads(f.read())
            self.timestep = foo["timestep"]
            self.batch_size = foo["batch_size"]
            os.environ["OMP_NUM_THREADS"] = foo["OMP_NUM_THREADS"]
            self.inter_op_parallelism_threads = foo["inter_op_parallelism_threads"]
            self.intra_op_parallelism_threads = foo["intra_op_parallelism_threads"]
            tf.config.threading.set_inter_op_parallelism_threads(self.inter_op_parallelism_threads)
            tf.config.threading.set_intra_op_parallelism_threads(self.intra_op_parallelism_threads)

parser = argparse.ArgumentParser("PyTorch Hyperparameter Sweep Test")
parser.add_argument("--ray-address", type=str, help="The Redis address of the cluster.")
parser.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
parser.add_argument("--num_samples", default=100, help="number of samples for ray tune")
args = parser.parse_args()

ray.init(address=args.ray_address, num_cpus=6 if args.smoke_test else None)

# validate_save_restore(MyTrainableClass)
# validate_save_restore(MyTrainableClass, use_object_store=True)


analysis = tune.run(MyTrainableClass,
    name="tf2_resnet_test",
    # scheduler=ahsa,
    stop={"training_iteration": 3 if args.smoke_test else 10},
    num_samples=1 if args.smoke_test else args.num_samples,
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    checkpoint_at_end=True,
    checkpoint_freq=3,
    config={
        "batch_size": tune.sample_from(lambda _: int(np.random.randint(1, high=64))),
        "OMP_NUM_THREADS": tune.sample_from(lambda _: int(np.random.randint(0, high=8))),
        # "KMP_BLOCKTIME": tune.sample_from(lambda _: int(100 * np.random.randint(0, high=10))),
        # "KMP_AFFINITY": tune.sample_from(lambda _: np.random.choice([""])),
        "inter_op_parallelism_threads": tune.sample_from(lambda _: int(np.random.randint(1, high=4))),
        "intra_op_parallelism_threads": 12,
        "warmup": True
    })

print("Best config is:", analysis.get_best_config(metric="throughput", mode="max"))
