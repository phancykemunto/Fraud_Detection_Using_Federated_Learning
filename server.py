import flwr as fl
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from flwr.common import Parameters, Scalar
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import parameters_to_ndarrays, Parameters

# Metric aggregation function
def aggregate_metrics(metrics):
    print("Metrics received:", metrics)
    accuracies = [m[1].get("accuracy", 0) for m in metrics]
    precisions = [m[1].get("precision", 0) for m in metrics]
    f1_scores = [m[1].get("f1_score", 0) for m in metrics]
    aucs = [m[1].get("auc", 0) for m in metrics]
    losses = [m[1].get("loss", 0) for m in metrics]
    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "f1_score": np.mean(f1_scores),
        "auc": np.mean(aucs),
        "loss": np.mean(losses)
    }

# Custom strategy class
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Optional[Tuple[Parameters, dict]]:
        # Call FedAvg's aggregate_fit, which returns a tuple: (parameters, metrics)
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        
        if aggregated_result is not None:
            aggregated_parameters, aggregated_metrics = aggregated_result

            # Save model weights
            weights = parameters_to_ndarrays(aggregated_parameters)
            with open("global_model_weights.pkl", "wb") as f:
                pickle.dump(weights, f)

            # Return both parameters and metrics (as expected)
            return aggregated_parameters, aggregated_metrics

        return None

# Initialize and start the server
strategy = SaveModelStrategy(
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=aggregate_metrics
)

fl.server.start_server(
    server_address="127.0.0.1:8082",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=5)
)