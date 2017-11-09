from keras.callbacks import Callback


class Schedule(Callback):
    """
    at the end of an epoch it will call the `schedule` function (giving the epoch number) on its list of variables
    """

    def __init__(self, variables, schedule_fn):
        super(Schedule, self).__init__()
        self.variables = variables
        if not isinstance(variables, list) or not isinstance(variables, tuple):
            self.variables = [variables]
        self.schedule_fn = schedule_fn

    def on_epoch_end(self, epoch, logs=None):
        for variable in self.variables:
            self.schedule_fn(variable, epoch)
