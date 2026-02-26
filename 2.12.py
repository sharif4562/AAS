import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

class OxfordRobotCarPlaceRecognizer:
    """
    Simulates place recognition for Oxford RobotCar dataset
    Uses image descriptors and similarity threshold for place matching
    """
    
    def __init__(self, similarity_threshold=0.9):
        self.threshold = similarity_threshold
        self.reference_descriptors = self._create_reference_database()
        
    def _create_reference_database(self):
        """
        Create synthetic reference descriptors representing places
        in the Oxford RobotCar route
        """
        np.random.seed(42)  # For reproducibility
        n_places = 50
        descriptor_dim = 256
        
        # Create reference descriptors for different locations
        references = {}
        for i in range(n_places):
            # Each place has a unique descriptor with some noise
            base_descriptor = np.random.randn(descriptor_dim)
            base_descriptor = base_descriptor / np.linalg.norm(base_descriptor)
            
            # Store multiple variations for each place (different weather/lighting)
            variations = []
            for j in range(5):
                variation = base_descriptor + 0.1 * np.random.randn(descriptor_dim)
                variation = variation / np.linalg.norm(variation)
                variations.append(variation)
            
            references[f"place_{i:03d}"] = {
                'base': base_descriptor,
                'variations': variations,
                'location': (51.75 + 0.001*i, -1.25 + 0.0005*i)  # Approx Oxford coordinates
            }
        
        return references
    
    def compute_similarity(self, descriptor1, descriptor2):
        """Compute cosine similarity between two descriptors"""
        return cosine_similarity([descriptor1], [descriptor2])[0][0]
    
    def recognize_place(self, query_descriptor):
        """
        Match query image to reference places based on similarity threshold
        Returns: (recognized, place_id, similarity_score)
        """
        best_match = None
        best_similarity = -1
        
        # Compare with all reference places
        for place_id, place_data in self.reference_descriptors.items():
            # Check against base descriptor and all variations
            all_refs = [place_data['base']] + place_data['variations']
            
            for ref_desc in all_refs:
                similarity = self.compute_similarity(query_descriptor, ref_desc)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = place_id
        
        # Apply threshold
        recognized = best_similarity > self.threshold
        
        return recognized, best_match, best_similarity

# Main execution
print("=" * 70)
print("GOOGLE - Google Maps: Oxford RobotCar Place Recognition")
print("=" * 70)

# Initialize recognizer
recognizer = OxfordRobotCarPlaceRecognizer(similarity_threshold=0.9)

print("\n📍 Oxford RobotCar Dataset Simulation")
print("   Route: Consistent loop through Oxford, UK")
print("   Conditions: Varying weather, traffic, seasons")
print(f"   Reference Places: {len(recognizer.reference_descriptors)} locations")

# Simulate query images from the dataset
print("\n🔍 Running Place Recognition:")

# Test Case 1: Known place (high similarity)
print("\n" + "-" * 50)
print("TEST CASE 1: Vehicle at previously visited location")

# Create query descriptor similar to an existing place
known_place_id = "place_025"
known_place_desc = recognizer.reference_descriptors[known_place_id]['base']
# Add small noise to simulate real query
query_desc = known_place_desc + 0.05 * np.random.randn(256)
query_desc = query_desc / np.linalg.norm(query_desc)

recognized, match_id, similarity = recognizer.recognize_place(query_desc)

print(f"   Query Similarity: {similarity:.3f}")
print(f"   Threshold: > {recognizer.threshold}")
print(f"   Match: {match_id}")

if recognized:
    print(f"   ✅ OUTPUT: Place Recognized")
    print(f"   Industry: Google Maps")
    print(f"\n   📊 Recognition Details:")
    print(f"      - Matched to: {match_id}")
    print(f"      - Location: {recognizer.reference_descriptors[match_id]['location']}")
    print(f"      - Confidence: {similarity*100:.1f}%")
else:
    print(f"   ❌ Not recognized")

# Test Case 2: Unknown place (low similarity)
print("\n" + "-" * 50)
print("TEST CASE 2: Vehicle at new/unseen location")

# Create random descriptor (simulating new place)
unknown_desc = np.random.randn(256)
unknown_desc = unknown_desc / np.linalg.norm(unknown_desc)

recognized, match_id, similarity = recognizer.recognize_place(unknown_desc)

print(f"   Query Similarity: {similarity:.3f}")
print(f"   Threshold: > {recognizer.threshold}")

if recognized:
    print(f"   ✅ Recognized as: {match_id}")
else:
    print(f"   ❌ No match above threshold")

# Test Case 3: Place with different condition (weather/lighting)
print("\n" + "-" * 50)
print("TEST CASE 3: Same location, different weather/lighting")

# Use a variation of the known place (simulating different conditions)
weather_variation = recognizer.reference_descriptors[known_place_id]['variations'][2]

recognized, match_id, similarity = recognizer.recognize_place(weather_variation)

print(f"   Query Similarity: {similarity:.3f}")
print(f"   Threshold: > {recognizer.threshold}")
print(f"   Match: {match_id}")

if recognized:
    print(f"   ✅ OUTPUT: Place Recognized")
    print(f"   Industry: Google Maps")
    print(f"\n   🌦️  Robust to Weather Changes:")
    print(f"      - Successfully matched despite different conditions")
    print(f"      - Similarity score: {similarity:.3f}")

# Summary statistics
print("\n" + "=" * 70)
print("📊 RECOGNITION STATISTICS")
print("=" * 70)

# Generate multiple test queries
n_tests = 100
known_ratio = 0.7  # 70% from known places, 30% novel

results = []
for i in range(n_tests):
    if random.random() < known_ratio:
        # Query from known place
        place_id = random.choice(list(recognizer.reference_descriptors.keys()))
        base_desc = recognizer.reference_descriptors[place_id]['base']
        # Add realistic noise
        query = base_desc + 0.08 * np.random.randn(256)
        query = query / np.linalg.norm(query)
        is_known = True
    else:
        # Novel location
        query = np.random.randn(256)
        query = query / np.linalg.norm(query)
        is_known = False
    
    recognized, _, sim = recognizer.recognize_place(query)
    results.append({
        'known': is_known,
        'recognized': recognized,
        'similarity': sim
    })

# Calculate metrics
true_positives = sum(1 for r in results if r['known'] and r['recognized'])
false_negatives = sum(1 for r in results if r['known'] and not r['recognized'])
false_positives = sum(1 for r in results if not r['known'] and r['recognized'])
true_negatives = sum(1 for r in results if not r['known'] and not r['recognized'])

print(f"\n   Total Tests: {n_tests}")
print(f"   Known Places: {sum(r['known'] for r in results)}")
print(f"   Novel Places: {sum(not r['known'] for r in results)}")
print(f"\n   True Positives: {true_positives} (known & recognized)")
print(f"   False Negatives: {false_negatives} (known but missed)")
print(f"   False Positives: {false_positives} (novel but claimed known)")
print(f"\n   Recognition Rate: {true_positives/(true_positives+false_negatives)*100:.1f}%")
print(f"   Precision: {true_positives/(true_positives+false_positives)*100:.1f}%")

print("\n" + "=" * 70)
print("📌 About Oxford RobotCar Dataset:")
print("   • Over 100 repetitions of same route through Oxford")
print("   • Captures weather, traffic, seasonal changes")
print("   • Ideal for long-term place recognition research")
print("   • Used in Google Maps for robust localization")
print("=" * 70)

# Optional: Visualize similarity distribution
try:
    plt.figure(figsize=(10, 6))
    
    known_sims = [r['similarity'] for r in results if r['known']]
    novel_sims = [r['similarity'] for r in results if not r['known']]
    
    plt.hist(known_sims, bins=20, alpha=0.7, label='Known Places', color='green')
    plt.hist(novel_sims, bins=20, alpha=0.7, label='Novel Places', color='red')
    plt.axvline(x=recognizer.threshold, color='blue', linestyle='--', 
                label=f'Threshold = {recognizer.threshold}')
    
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Place Recognition Similarity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('place_recognition_results.png')
    print(f"\n✅ Visualization saved as 'place_recognition_results.png'")
except Exception as e:
    print(f"\n⚠️ Visualization skipped: {e}")
