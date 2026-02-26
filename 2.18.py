import pandas as pd
import numpy as np
from itertools import product

class RuleBasedAgent:
    """
    IBM-style rule-based agent for logical decision making
    Implements if-then rules to control system behavior
    """
    
    def __init__(self):
        self.rules = []
        self.facts = {}
        self.actions = []
        
    def add_rule(self, condition, action):
        """
        Add a logical rule: if condition then action
        
        Args:
            condition: dict of {attribute: value} pairs that must be true
            action: action to take when condition is met
        """
        self.rules.append({
            'condition': condition,
            'action': action
        })
        
    def set_fact(self, attribute, value):
        """Set a known fact in the knowledge base"""
        self.facts[attribute] = value
        
    def evaluate(self):
        """
        Evaluate all rules against current facts
        Returns first matching action
        """
        print("🔍 Evaluating rules against facts:")
        print(f"   Facts: {self.facts}")
        
        for i, rule in enumerate(self.rules):
            print(f"\n   Checking Rule {i+1}:")
            print(f"      If {rule['condition']}")
            
            # Check if all conditions match
            match = True
            for attr, required_value in rule['condition'].items():
                actual_value = self.facts.get(attr)
                print(f"         {attr} = {actual_value} (required: {required_value})")
                
                if actual_value != required_value:
                    match = False
                    print(f"         ❌ Condition failed")
                    break
                else:
                    print(f"         ✅ Condition met")
            
            if match:
                print(f"      ✅ ALL CONDITIONS MET → Action: {rule['action']}")
                return rule['action']
            else:
                print(f"      ❌ Rule does not apply")
        
        print("\n   No matching rules found")
        return None
    
    def run_inference(self):
        """Run forward chaining inference"""
        action = self.evaluate()
        
        if action:
            self.actions.append(action)
            print(f"\n🎯 OUTPUT: {action}")
            print(f"   Industry: IBM AI Operations")
            return action
        else:
            print("\n🎯 OUTPUT: No action taken")
            print(f"   Industry: IBM AI Operations")
            return None

def create_sample_rule_dataset():
    """
    Create a sample dataset mimicking UCI Rule Learning dataset format
    Based on typical rule learning problems (like tic-tac-toe, car evaluation)
    """
    # Generate synthetic rule learning data
    np.random.seed(42)
    
    # Define attributes and possible values
    attributes = {
        'temperature': ['low', 'medium', 'high'],
        'pressure': ['low', 'normal', 'high'],
        'humidity': ['dry', 'normal', 'wet'],
        'motion': ['stopped', 'moving', 'accelerating'],
        'distance': ['near', 'far', 'very_far']
    }
    
    # Generate random samples
    n_samples = 100
    data = []
    
    for _ in range(n_samples):
        sample = {}
        for attr, values in attributes.items():
            sample[attr] = np.random.choice(values)
        
        # Generate label based on logical rules
        if (sample['temperature'] == 'high' and 
            sample['pressure'] == 'high' and 
            sample['distance'] == 'near'):
            label = 'stop'
        elif (sample['motion'] == 'stopped' and 
              sample['distance'] == 'near'):
            label = 'stop'
        elif (sample['temperature'] == 'low' and 
              sample['humidity'] == 'wet'):
            label = 'caution'
        else:
            label = 'continue'
        
        sample['label'] = label
        data.append(sample)
    
    return pd.DataFrame(data), attributes

# Main execution
print("=" * 70)
print("IBM - Rule-Based Agent: Logical Rules Dataset")
print("=" * 70)

# Create sample dataset (simulating UCI Rule Learning dataset)
print("\n📊 Generating Synthetic Rule Learning Dataset...")
dataset, attr_domains = create_sample_rule_dataset()

print(f"\nDataset Overview:")
print(f"   • {len(dataset)} samples")
print(f"   • Attributes: {', '.join(attr_domains.keys())}")
print(f"   • Labels: {dataset['label'].unique()}")

# Display sample data
print("\n📋 First 5 samples:")
print(dataset.head().to_string())

# Define rules based on domain knowledge
print("\n" + "-" * 70)
print("⚙️ Defining Logical Rules:")

agent = RuleBasedAgent()

# Rule 1: High temperature + high pressure + near distance → STOP
agent.add_rule(
    condition={
        'temperature': 'high',
        'pressure': 'high',
        'distance': 'near'
    },
    action='stop'
)
print("   Rule 1: IF temperature=high AND pressure=high AND distance=near THEN action=stop")

# Rule 2: Stopped motion + near distance → STOP
agent.add_rule(
    condition={
        'motion': 'stopped',
        'distance': 'near'
    },
    action='stop'
)
print("   Rule 2: IF motion=stopped AND distance=near THEN action=stop")

# Rule 3: Low temperature + wet humidity → CAUTION
agent.add_rule(
    condition={
        'temperature': 'low',
        'humidity': 'wet'
    },
    action='caution'
)
print("   Rule 3: IF temperature=low AND humidity=wet THEN action=caution")

# Rule 4: Default action
agent.add_rule(
    condition={},  # Empty condition means always true
    action='continue'
)
print("   Rule 4: DEFAULT → action=continue")

# Test the rule agent with various scenarios
print("\n" + "-" * 70)
print("🧪 Testing Rule Agent with Different Scenarios:")
print("-" * 70)

# Test Case 1: Emergency stop scenario
print("\n📌 TEST CASE 1: Emergency Situation")
agent.set_fact('temperature', 'high')
agent.set_fact('pressure', 'high')
agent.set_fact('humidity', 'normal')
agent.set_fact('motion', 'accelerating')
agent.set_fact('distance', 'near')

agent.run_inference()

# Test Case 2: Stationary obstacle
print("\n" + "-" * 70)
print("\n📌 TEST CASE 2: Stationary Obstacle Nearby")
agent.set_fact('temperature', 'normal')
agent.set_fact('pressure', 'normal')
agent.set_fact('humidity', 'normal')
agent.set_fact('motion', 'stopped')
agent.set_fact('distance', 'near')

agent.run_inference()

# Test Case 3: Normal operation
print("\n" + "-" * 70)
print("\n📌 TEST CASE 3: Normal Operation")
agent.set_fact('temperature', 'normal')
agent.set_fact('pressure', 'normal')
agent.set_fact('humidity', 'dry')
agent.set_fact('motion', 'moving')
agent.set_fact('distance', 'far')

agent.run_inference()

# Test Case 4: Weather caution
print("\n" + "-" * 70)
print("\n📌 TEST CASE 4: Adverse Weather")
agent.set_fact('temperature', 'low')
agent.set_fact('pressure', 'normal')
agent.set_fact('humidity', 'wet')
agent.set_fact('motion', 'moving')
agent.set_fact('distance', 'far')

agent.run_inference()

# Analyze rule coverage
print("\n" + "=" * 70)
print("📊 Rule Coverage Analysis")
print("=" * 70)

# Test all possible combinations (simplified)
print("\nTesting rule coverage on dataset:")
rule_matches = {f"Rule {i+1}": 0 for i in range(len(agent.rules))}

for idx, row in dataset.iterrows():
    # Set facts from dataset
    for attr in attr_domains.keys():
        agent.set_fact(attr, row[attr])
    
    # Find which rule would fire
    action = None
    for i, rule in enumerate(agent.rules):
        match = True
        for attr, val in rule['condition'].items():
            if row[attr] != val:
                match = False
                break
        if match:
            rule_matches[f"Rule {i+1}"] += 1
            break

print("\nRule application frequency:")
for rule, count in rule_matches.items():
    percentage = (count / len(dataset)) * 100
    print(f"   {rule}: {count} times ({percentage:.1f}%)")

# Confusion matrix for rule-based predictions
print("\n📈 Prediction Analysis:")
predictions = []
actuals = []

for idx, row in dataset.iterrows():
    # Set facts
    for attr in attr_domains.keys():
        agent.set_fact(attr, row[attr])
    
    # Get prediction
    action = None
    for rule in agent.rules:
        match = True
        for attr, val in rule['condition'].items():
            if row[attr] != val:
                match = False
                break
        if match:
            action = rule['action']
            break
    
    predictions.append(action)
    actuals.append(row['label'])

# Calculate accuracy
accuracy = sum(1 for p, a in zip(predictions, actuals) if p == a) / len(actuals)
print(f"\n   Rule-based system accuracy on dataset: {accuracy:.1%}")

print("\n" + "=" * 70)
print("📌 About Rule Learning Datasets:")
print("   • UCI ML Repository contains many rule learning datasets")
print("   • Examples: tic-tac-toe, car evaluation, mushroom classification")
print("   • Rules are typically of form: IF conditions THEN class")
print("   • Used in expert systems and decision support")
print("=" * 70)

# How to access real rule learning datasets
print("\n🔍 Alternative ways to access rule learning datasets:")
print("   1. UCI ML Repository main page: https://archive.ics.uci.edu/")
print("      Search for: 'tic-tac-toe', 'car evaluation', 'nursery'")
print("   2. OpenML: https://www.openml.org/search?type=data&sort=runs&status=active")
print("   3. Kaggle Datasets: https://www.kaggle.com/datasets")
print("   4. PMLB (Penn Machine Learning Benchmarks): https://github.com/EpistasisLab/pmlb")

# Example of loading real UCI data
print("\n📥 Example of loading real UCI dataset:")
print("```python")
print("import pandas as pd")
print("import urllib.request")
print("")
print("# Example: Car Evaluation Dataset")
print("url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'")
print("columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']")
print("df = pd.read_csv(url, names=columns)")
print("print(df.head())")
print("```")
