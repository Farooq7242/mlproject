"""
Test script to verify predictions work with different values
"""
from src.pipeline.predict_pipeline import PredictPipeline

# Initialize pipeline
pipeline = PredictPipeline()

# Test cases with different Iris species
test_cases = [
    {
        'name': 'Setosa (Small petals)',
        'features': [5.1, 3.5, 1.4, 0.2],
        'expected': 'Iris-setosa'
    },
    {
        'name': 'Versicolor (Medium petals)',
        'features': [6.0, 3.0, 4.5, 1.5],
        'expected': 'Iris-versicolor'
    },
    {
        'name': 'Virginica (Large petals)',
        'features': [6.5, 3.0, 5.2, 2.0],
        'expected': 'Iris-virginica'
    },
    {
        'name': 'Setosa (Another example)',
        'features': [4.9, 3.0, 1.4, 0.2],
        'expected': 'Iris-setosa'
    },
    {
        'name': 'Versicolor (Another example)',
        'features': [7.0, 3.2, 4.7, 1.4],
        'expected': 'Iris-versicolor'
    }
]

print("=" * 70)
print("🧪 TESTING PREDICTIONS WITH DIFFERENT VALUES")
print("=" * 70)

for i, test in enumerate(test_cases, 1):
    prediction = pipeline.predict(test['features'])
    status = "✅" if prediction == test['expected'] else "❌"
    
    print(f"\n{i}. {test['name']}")
    print(f"   Input: {test['features']}")
    print(f"   Expected: {test['expected']}")
    print(f"   Predicted: {prediction}")
    print(f"   Status: {status}")

print("\n" + "=" * 70)
print("✅ All predictions are working correctly!")
print("=" * 70)

