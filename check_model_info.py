"""
Script to check which model is currently being used and view model comparison results
"""
import os
import pickle
import json
from datetime import datetime

def check_current_model():
    """Check which model is currently saved and being used"""
    model_path = os.path.join('artifacts', 'model.pkl')
    comparison_path = os.path.join('artifacts', 'model_comparison.json')
    
    print("=" * 70)
    print("📊 MODEL INFORMATION")
    print("=" * 70)
    
    # Check if model exists
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            model_type = type(model).__name__
            print(f"\n✅ Current Model in Use: {model_type}")
            print(f"📍 Location: {model_path}")
            
            # Try to get model details
            if hasattr(model, 'get_params'):
                print(f"\n📋 Model Parameters:")
                params = model.get_params()
                for key, value in list(params.items())[:5]:  # Show first 5 params
                    print(f"   - {key}: {value}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
    else:
        print(f"\n❌ Model file not found at: {model_path}")
        print("   Please run training first: python train_models.py")
    
    # Check if comparison results exist
    print("\n" + "-" * 70)
    if os.path.exists(comparison_path):
        try:
            with open(comparison_path, 'r') as f:
                comparison = json.load(f)
            
            print("📈 MODEL COMPARISON RESULTS:")
            print("-" * 70)
            
            if 'best_model' in comparison:
                print(f"\n🏆 Best Model: {comparison['best_model']}")
                print(f"   Accuracy: {comparison['best_accuracy']:.4f} ({comparison['best_accuracy']*100:.2f}%)")
            
            if 'all_models' in comparison:
                print(f"\n📊 All Models Performance:")
                sorted_models = sorted(comparison['all_models'].items(), 
                                     key=lambda x: x[1], reverse=True)
                for i, (name, acc) in enumerate(sorted_models, 1):
                    marker = "🏆" if name == comparison.get('best_model') else "  "
                    print(f"{marker} {i}. {name}: {acc:.4f} ({acc*100:.2f}%)")
            
            if 'training_date' in comparison:
                print(f"\n📅 Training Date: {comparison['training_date']}")
                
        except Exception as e:
            print(f"❌ Error reading comparison file: {str(e)}")
    else:
        print("📈 MODEL COMPARISON RESULTS:")
        print("-" * 70)
        print("❌ Comparison results not found.")
        print("   Run training to generate comparison: python train_models.py")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_current_model()

