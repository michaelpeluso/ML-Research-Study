import time
import numpy as np
import psutil
import platform

def get_model_info(pipeline):
    model = pipeline.named_steps["estimator"]
    model_type = type(model).__name__
    info = {"model_type": model_type}
    
    if model_type in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
        info.update({
            "tree_depth": getattr(model, "get_depth", lambda: "N/A")(),
            "n_leaves": getattr(model, "get_n_leaves", lambda: "N/A")(),
            "max_features": getattr(model, "max_features", "N/A"),
            "criterion": getattr(model, "criterion", "N/A")
        })
    elif model_type in ["KNeighborsClassifier", "KNeighborsRegressor"]:
        info.update({
            "n_neighbors": getattr(model, "n_neighbors", "N/A"),
            "algorithm": getattr(model, "algorithm", "N/A"),
            "metric": getattr(model, "metric", "N/A")
        })
    elif model_type in ["SVC", "SVR"]:
        info.update({
            "kernel": getattr(model, "kernel", "N/A"),
            "C": getattr(model, "C", "N/A"),
            "gamma": getattr(model, "gamma", "N/A")
        })
        if hasattr(model, "support_vectors_") and hasattr(model, "n_support_"):
            total_samples = sum(model.n_support_) if hasattr(model, "n_support_") else "N/A"
            info["support_vector_fraction"] = total_samples / len(model.support_vectors_) if total_samples != "N/A" else "N/A"
    elif model_type in ["SGDClassifier", "SGDRegressor"]:
        info.update({
            "alpha": getattr(model, "alpha", "N/A"),
            "loss": getattr(model, "loss", "N/A"),
            "learning_rate": getattr(model, "learning_rate", "N/A")
        })
        if hasattr(model, "coef_"):
            n_support = np.sum(model.coef_ != 0) / model.coef_.size
            info["support_vector_fraction"] = n_support
    elif model_type in ["MLPClassifier", "MLPRegressor"]:
        info.update({
            "hidden_layers": getattr(model, "hidden_layer_sizes", "N/A"),
            "activation": getattr(model, "activation", "N/A"),
            "solver": getattr(model, "solver", "N/A"),
            "learning_rate_init": getattr(model, "learning_rate_init", "N/A")
        })
        if hasattr(model, "coefs_"):
            param_count = sum(np.prod(l.shape) for l in model.coefs_ + model.intercepts_)
            info["n_params"] = param_count
        if hasattr(model, "n_layers_"):
            info["n_layers"] = getattr(model, "n_layers_", "N/A")
    
    return info

def print_model_info(pipeline):
    model = pipeline.named_steps["estimator"]
    model_type = type(model).__name__
    print(f"Estimator type: {model_type}")

    if model_type in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
        print("Tree depth:", getattr(model, "get_depth", lambda: "N/A")())
        print("Number of leaves:", getattr(model, "get_n_leaves", lambda: "N/A")())
        print("Max features:", getattr(model, "max_features", "N/A"))
        print("Criterion:", getattr(model, "criterion", "N/A"))
    elif model_type in ["KNeighborsClassifier", "KNeighborsRegressor"]:
        print("Number of neighbors:", getattr(model, "n_neighbors", "N/A"))
        print("Algorithm:", getattr(model, "algorithm", "N/A"))
        print("Metric:", getattr(model, "metric", "N/A"))
    elif model_type in ["SGDClassifier", "SGDRegressor"]:
        print("Kernel:", getattr(model, "kernel", "N/A"))
        print("C:", getattr(model, "C", "N/A"))
        print("Alpha:", getattr(model, "alpha", "N/A"))
        if hasattr(model, "probability"):
            print("Probability enabled:", model.probability)
    elif model_type in ["SVC", "SVR"]:
        print("C:", getattr(model, "C", "N/A"))
        print("Gamma:", getattr(model, "gamma", "N/A"))
        if hasattr(model, "probability"):
            print("Probability enabled:", model.probability)
    elif model_type in ["MLPClassifier", "MLPRegressor"]:
        print("Hidden layers:", getattr(model, "hidden_layer_sizes", "N/A"))
        if hasattr(model, "n_outputs_"):
            print("Number of outputs:", getattr(model, "n_outputs_", "N/A"))
        if hasattr(model, "n_layers_"):
            print("Number of layers:", getattr(model, "n_layers_", "N/A"))
        print("Activation:", getattr(model, "activation", "N/A"))
        print("Solver:", getattr(model, "solver", "N/A"))
        if hasattr(model, "coefs_"):
            param_count = sum(np.prod(l.shape) for l in model.coefs_ + model.intercepts_)
            print("Total trainable params:", param_count)
    else:
        print("No specific info available for this model.")

def measure_runtime(pipeline, X_train, y_train, X_test):
    print("Measuring model run time.")
    start = time.time()
    pipeline.fit(X_train, y_train)
    fit_time = time.time() - start
    
    start = time.time()
    pipeline.predict(X_test)
    predict_time = time.time() - start
    
    return {"fit_time_seconds": fit_time, "predict_time_seconds": predict_time}

def get_hardware_info():
    return {
        "cpu_cores": psutil.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": platform.system()
    }