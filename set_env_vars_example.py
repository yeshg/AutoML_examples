import os, json, argparse, random

import numpy as np

import ray
import ray.tune as tune
from ray.tune import Trainable
from ray.tune.utils import validate_save_restore
from ray.tune.schedulers import HyperBandScheduler

class MyTrainableClass(Trainable):
    """Example agent whose learning curve is a random sigmoid.
    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def _setup(self, config):
        self.timestep = 0
        os.environ["TESTVAR"] = str(config["TESTVAR"])
        print("{}: {}".format(self.trial_id, os.environ.get("TESTVAR")))

    def _train(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config.get("width", 1))
        v *= self.config.get("height", 1)

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        return {"episode_reward_mean": v}

    def _save(self, checkpoint_dir):
        print("{}: {}".format(self.trial_id, os.environ.get("TESTVAR")))
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep, "TESTVAR": os.environ.get("TESTVAR")}))
        return path

    def _restore(self, checkpoint_path):
        with open(checkpoint_path) as f:
            foo = json.loads(f.read())
            self.timestep = foo["timestep"]
            print("{}: {}".format(self.trial_id, foo["TESTVAR"]))

parser = argparse.ArgumentParser("Ray Tune Environment Variables Test")
parser.add_argument("--smoke-test", default=False, action="store_true", help="Finish quickly for testing")
parser.add_argument("--ray-address", default=None, help="Address of Ray cluster for seamless distributed execution.")
args, _ = parser.parse_known_args()
ray.init(address=args.ray_address) if args.ray_address is not None else ray.init()

# validate_save_restore(MyTrainableClass)
# validate_save_restore(MyTrainableClass, use_object_store=True)

hb = HyperBandScheduler(
    metric="episode_reward_mean",
    mode="max")

tune.run(MyTrainableClass,
    name="asynchyperband_test",
    scheduler=hb,
    stop={"training_iteration": 1 if args.smoke_test else 99999},
    num_samples=20,
    resources_per_trial={
        "cpu": 1,
        "gpu": 0
    },
    config={
        "width": tune.sample_from(lambda spec: 10 + int(90 * random.random())),
        "height": tune.sample_from(lambda spec: int(100 * random.random())),
        "TESTVAR": tune.sample_from(lambda spec: int(100 * random.random()))
    })