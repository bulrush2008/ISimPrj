class MetricTracker:
    def __init__(self):
        self.reset()

    def add(self, *values):
        if not values:
            raise ValueError("At least one value must be provided")
        
        for value in values:
            try:
                v = float(value)  # ensure scalar
                self.values.append(v)
                self.total += v
                self.count += 1
                self.max_val = max(self.max_val, v)
                self.min_val = min(self.min_val, v)
            except (TypeError, ValueError) as e:
                raise ValueError(f"All values must be numbers, got: {type(value)}") from e

    def average(self):
        return self.total / self.count if self.count != 0 else 0.0

    def std(self):
        if self.count < 2:
            return 0.0
        mean = self.average()
        return (sum((x - mean) ** 2 for x in self.values) / (self.count - 1)) ** 0.5

    def max(self):
        return self.max_val

    def min(self):
        return self.min_val
    
    def summary(self):
        return {
            "avg": self.average(),
            "std": self.std(),
            "max": self.max(),
            "min": self.min()
        }

    def reset(self):
        self.values = []
        self.total = 0.0
        self.count = 0
        self.max_val = float("-inf")
        self.min_val = float("inf")