import sys
sys.path.append("../src")
import os
import torch
import numpy as np
import pickle
from interpret.glassbox import ExplainableBoostingClassifier
from sae_encoding import encode_outputs, encode_mechsparse_outputs, sae_encoding
from utils import compute_mutual_information_chunked, compute_conditional_mutual_information_chunked

class RAGLens:

    def __init__(
        self, 
        tokenizer, 
        model, 
        sae, 
        hookpoint, 
        copy_heads=None,
        knowledge_layers=None,
        use_mechsparse: bool = False,
        top_k = 1000,
        max_bins = 32, 
        random_state = 0,
        early_stopping_tolerance = 1e-5,
        validation_size = 0.1,
        max_rounds = 1000,
    ):
        
        self.tokenizer = tokenizer
        self.model = model
        self.sae = sae
        self.hookpoint = hookpoint
        self.copy_heads = copy_heads
        self.knowledge_layers = knowledge_layers
        self.use_mechsparse = use_mechsparse
        self.clf = ExplainableBoostingClassifier(
            interactions = 0, 
            max_bins = max_bins, 
            random_state = random_state, 
            early_stopping_tolerance = early_stopping_tolerance, 
            validation_size = validation_size, 
            max_rounds = max_rounds
        )
        self.top_k = top_k
        self.top_k_indices = None
        self._trained_with_H = False
        self._h_mean = None

    def save(self, path: str):
        """
        Save the trained detector (EBM + selected feature indices + MechSparse circuit params).
        Note: SAE/model/tokenizer are NOT serialized.
        """
        obj = {
            "top_k": self.top_k,
            "top_k_indices": None if self.top_k_indices is None else self.top_k_indices.tolist(),
            "hookpoint": self.hookpoint,
            "use_mechsparse": self.use_mechsparse,
            "copy_heads": self.copy_heads,
            "knowledge_layers": self.knowledge_layers,
            "clf": self.clf,
            "_trained_with_H": self._trained_with_H,
            "_h_mean": self._h_mean,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    @classmethod
    def load(cls, path: str, *, tokenizer, model, sae):
        """
        Load detector; caller supplies tokenizer/model/sae runtime objects.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        inst = cls(
            tokenizer=tokenizer,
            model=model,
            sae=sae,
            hookpoint=obj["hookpoint"],
            copy_heads=obj.get("copy_heads"),
            knowledge_layers=obj.get("knowledge_layers"),
            use_mechsparse=bool(obj.get("use_mechsparse", False)),
            top_k=int(obj.get("top_k", 1000)),
        )
        inst.top_k_indices = None if obj.get("top_k_indices") is None else np.array(obj["top_k_indices"], dtype=int)
        inst.clf = obj["clf"]
        inst._trained_with_H = bool(obj.get("_trained_with_H", False))
        inst._h_mean = obj.get("_h_mean", None)
        return inst

    def fit(
        self,
        inputs,
        outputs,
        labels,
        H_values=None,
        save_path=None,
        n_bins = 50,
        chunk_size = 2000,
        conditional_mi = False,
        n_cond_bins = 8,
    ):

        assert type(inputs) == type(outputs) == type(labels) == list
        assert len(inputs) == len(outputs) == len(labels)
        if conditional_mi:
            assert H_values is not None, "conditional_mi=True requires H_values aligned with samples"
            assert type(H_values) == list and len(H_values) == len(labels)
        self._trained_with_H = H_values is not None
        if self._trained_with_H:
            self._h_mean = float(np.mean(np.array(H_values, dtype=np.float32)))
        
        if save_path and os.path.exists(save_path):
            features = np.load(save_path)
        else:
            print("Encoding training features with SAE...")
            if self.use_mechsparse:
                assert self.copy_heads is not None and self.knowledge_layers is not None, "MechSparse requires copy_heads and knowledge_layers"
                features = encode_mechsparse_outputs(
                    inputs,
                    outputs,
                    self.tokenizer,
                    self.model,
                    self.sae,
                    copy_heads=self.copy_heads,
                    knowledge_layers=self.knowledge_layers,
                    agg="max",
                    show_progress=True,
                ).numpy()
            else:
                features = encode_outputs(
                    inputs, 
                    outputs, 
                    self.hookpoint, 
                    self.tokenizer, 
                    self.model, 
                    self.sae, 
                    agg='max', 
                    show_progress=True
                ).numpy()
            if save_path:
                np.save(save_path, features)

        # Mutual Information-based Feature Selection
        print(f"Extracting top {self.top_k} key SAE features...")
        if conditional_mi:
            mi_scores = compute_conditional_mutual_information_chunked(
                torch.tensor(features),
                torch.tensor(labels),
                torch.tensor(H_values),
                n_bins=n_bins,
                n_cond_bins=n_cond_bins,
                chunk_size=chunk_size,
            ).cpu().numpy()
        else:
            mi_scores = compute_mutual_information_chunked(
                torch.tensor(features),
                torch.tensor(labels),
                n_bins=n_bins,
                chunk_size=chunk_size
            ).cpu().numpy()

        sorted_indices = np.argsort(mi_scores)[::-1]
        self.top_k_indices = sorted_indices[:self.top_k]

        # Generalized Additive Model Fitting
        print(f"Fitting an additive model based on key SAE features...")
        X = features[:, self.top_k_indices]
        if H_values is not None:
            X = np.concatenate([X, np.array(H_values, dtype=np.float32).reshape(-1, 1)], axis=1)
        self.clf.fit(X, labels)

    def predict_proba(self, inputs, outputs, H_values=None):
        
        if type(inputs) == type(outputs) == str:
            inputs = [inputs]
            outputs = [outputs]
        
        assert type(inputs) == type(outputs) == list

        if self.use_mechsparse:
            assert self.copy_heads is not None and self.knowledge_layers is not None, "MechSparse requires copy_heads and knowledge_layers"
            features = encode_mechsparse_outputs(
                inputs,
                outputs,
                self.tokenizer,
                self.model,
                self.sae,
                copy_heads=self.copy_heads,
                knowledge_layers=self.knowledge_layers,
                agg="max",
                show_progress=False,
            ).numpy()[:, self.top_k_indices]
        else:
            features = encode_outputs(
                inputs, 
                outputs, 
                self.hookpoint, 
                self.tokenizer, 
                self.model, 
                self.sae, 
                agg='max', 
                show_progress=False
            ).numpy()[:, self.top_k_indices]
        X = features
        if self._trained_with_H:
            # If caller doesn't provide H online, fall back to training mean.
            if H_values is None:
                assert self._h_mean is not None, "Detector trained with H but missing internal _h_mean"
                H_fill = np.full((X.shape[0], 1), self._h_mean, dtype=np.float32)
                X = np.concatenate([X, H_fill], axis=1)
            else:
                assert len(H_values) == len(inputs)
                X = np.concatenate([X, np.array(H_values, dtype=np.float32).reshape(-1, 1)], axis=1)
        logits = self.clf.predict_proba(X)[:, 1]
        return logits
        
    def predict(self, inputs, outputs, H_values=None):

        logits = self.predict_proba(inputs, outputs, H_values=H_values)
        preds = (logits > 0.5).astype(int)

        return preds