<h1 align="center">G.O.D Subnet</h1>


🚀 Welcome to the [Gradients on Demand](https://gradients.io) Subnet

> Providing access to Bittensor network for on-demand training at scale.


## Setup Guides

- [Miner Setup Guide](docs/miner_setup.md)
- [Validator Setup Guide](docs/validator_setup.md)

## Recommended Compute Requirements

[Compute Requirements](docs/compute.md)

## Miner Advice

[Miner Advice](docs/miner_advice.md)


/root/G.O.D-test/axolotl/src/axolotl/core/trainer_builder.py
    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)
        if "ignore_patterns" not in kwargs:
            kwargs["ignore_patterns"] = ["README.md"]
        else:
            kwargs["ignore_patterns"].extend(["README.md"])
        
        kwargs["create_model_card"] = False  
        return super().push_to_hub(*args, **kwargs)
