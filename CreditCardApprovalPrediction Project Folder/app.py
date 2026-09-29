import os
import datetime
import pandas as pd
from flask import Flask, render_template, request

# Import loaders
from utils.model_loader import ModelRegistry

app = Flask(__name__)

# Initialize Model Registry at application startup
# This executes the loading and compatibility checks exactly once
registry = ModelRegistry()
try:
    registry.initialize()
except Exception as e:
    print(f"[FATAL] Model Registry initialization failed on startup: {str(e)}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Verify Model Registry status
    if not registry.initialized:
        return render_template('predict.html', 
                               error_alert="System Initialization Error: The eligibility assessment system could not be initialized. Please try again later or contact customer support.")

    if request.method == 'POST':
        try:
            # 1. Collect and clean inputs
            form_data = {key: value.strip() for key, value in request.form.items()}
            
            # List of required fields
            required_fields = [
                'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'AGE_YEARS', 
                'EMPLOYMENT_YEARS', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
                'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE', 'REGION_RATING_CLIENT', 
                'AMT_REQ_CREDIT_BUREAU_YEAR'
            ]
            
            # Check presence
            for field in required_fields:
                if field not in form_data or form_data[field] == '':
                    raise ValueError(f"Required field '{field.replace('_', ' ').title()}' is missing.")
            
            # 2. Server-Side Validations & Conversions
            # Convert and check Age (18 - 100)
            try:
                age = float(form_data['AGE_YEARS'])
                if age < 18 or age > 100:
                    raise ValueError("Age must be between 18 and 100.")
            except ValueError as e:
                raise ValueError(f"Invalid Age: {str(e) if 'between' in str(e) else 'Must be a valid number.'}")

            # Convert and check Employment Years (0 - 60)
            try:
                emp_years = float(form_data['EMPLOYMENT_YEARS'])
                if emp_years < 0 or emp_years > 60:
                    raise ValueError("Employment duration must be between 0 and 60 years.")
            except ValueError as e:
                raise ValueError(f"Invalid Employment Years: {str(e) if 'between' in str(e) else 'Must be a valid number.'}")

            # Convert and check Annual Income (> 0)
            try:
                income = float(form_data['AMT_INCOME_TOTAL'])
                if income <= 0:
                    raise ValueError("Annual income must be greater than zero.")
            except ValueError:
                raise ValueError("Invalid Annual Income: Must be a number greater than zero.")

            # Convert and check Credit Amount (> 0)
            try:
                credit = float(form_data['AMT_CREDIT'])
                if credit <= 0:
                    raise ValueError("Credit limit must be greater than zero.")
            except ValueError:
                raise ValueError("Invalid Credit Limit: Must be a number greater than zero.")

            # Convert and check Loan Annuity (> 0)
            try:
                annuity = float(form_data['AMT_ANNUITY'])
                if annuity <= 0:
                    raise ValueError("Loan annuity must be greater than zero.")
            except ValueError:
                raise ValueError("Invalid Loan Annuity: Must be a number greater than zero.")

            # Convert and check Children (0 - 20)
            try:
                children = int(form_data['CNT_CHILDREN'])
                if children < 0 or children > 20:
                    raise ValueError("Number of children must be between 0 and 20.")
            except ValueError as e:
                raise ValueError(f"Invalid Children Count: {str(e) if 'between' in str(e) else 'Must be an integer.'}")

            # Convert and check Family Members (1 - 20)
            try:
                fam_members = int(form_data['CNT_FAM_MEMBERS'])
                if fam_members < 1 or fam_members > 20:
                    raise ValueError("Family size must be between 1 and 20.")
                if fam_members < (children + 1):
                    raise ValueError(f"Family size must be at least {children + 1} (Children count + 1 applicant).")
            except ValueError as e:
                raise ValueError(f"Invalid Family Size: {str(e) if 'between' in str(e) else 'Must be an integer.'}")

            # Convert and check Credit Bureau Requests (>= 0)
            try:
                bureau_req = int(form_data['AMT_REQ_CREDIT_BUREAU_YEAR'])
                if bureau_req < 0:
                    raise ValueError("Credit bureau queries must be 0 or greater.")
            except ValueError as e:
                raise ValueError(f"Invalid Credit Bureau Queries: {str(e) if 'greater' in str(e) else 'Must be an integer.'}")

            # Convert and check Region Rating (Dropdown validation)
            try:
                region_rating = int(form_data['REGION_RATING_CLIENT'])
                if region_rating not in [1, 2, 3]:
                    raise ValueError("Region rating must be 1, 2, or 3.")
            except ValueError:
                raise ValueError("Invalid Region Rating selection.")

            # 3. Categorical Encodings & Column Construction
            features_dict = {}
            
            # Map elements
            categorical_cols = [
                'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 
                'NAME_HOUSING_TYPE', 'ORGANIZATION_TYPE'
            ]
            
            # Map and encode categorical columns
            for col in categorical_cols:
                val = form_data[col]
                encoder = registry.encoders.get(col)
                if encoder is None:
                    raise ValueError(f"System Configuration Error: Encoder for '{col}' is missing.")
                
                if val in encoder.classes_:
                    features_dict[col] = encoder.transform([val])[0]
                elif val + ' ' in encoder.classes_: # Handle minor spacing mismatches if any
                    features_dict[col] = encoder.transform([val + ' '])[0]
                else:
                    raise ValueError(f"Category '{val}' is invalid for input field '{col.replace('_', ' ').title()}'.")
            
            # Map raw numerical columns
            basic_numerical_cols = [
                'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
                'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'REGION_RATING_CLIENT',
                'AMT_REQ_CREDIT_BUREAU_YEAR'
            ]
            for col in basic_numerical_cols:
                features_dict[col] = float(form_data[col])
            
            # Clean employment years (cap negative values to 0.0)
            if features_dict['EMPLOYMENT_YEARS'] < 0:
                features_dict['EMPLOYMENT_YEARS'] = 0.0
                
            # Engineer credit risk ratio features dynamically
            features_dict['ANNUITY_TO_INCOME_RATIO'] = features_dict['AMT_ANNUITY'] / (features_dict['AMT_INCOME_TOTAL'] + 1e-5)
            features_dict['INCOME_TO_CREDIT_RATIO'] = features_dict['AMT_INCOME_TOTAL'] / (features_dict['AMT_CREDIT'] + 1e-5)
            features_dict['CREDIT_TO_ANNUITY_RATIO'] = features_dict['AMT_CREDIT'] / (features_dict['AMT_ANNUITY'] + 1e-5)
            features_dict['INCOME_PER_FAMILY_MEMBER'] = features_dict['AMT_INCOME_TOTAL'] / (features_dict['CNT_FAM_MEMBERS'] + 1e-5)
            
            # 4. Construct input DataFrame and Reorder Columns exactly
            input_df = pd.DataFrame([features_dict])
            
            # Safety checks for columns and feature count matching
            if len(input_df.columns) != len(registry.feature_columns):
                raise ValueError(f"Internal Feature Alignment Error: Constructed input shape has {len(input_df.columns)} columns, but registry expects {len(registry.feature_columns)}.")
                
            input_df = input_df[registry.feature_columns]

            # 5. Scaling Numerical Columns
            numerical_cols = [col for col in registry.feature_columns if col not in categorical_cols]
            input_df[numerical_cols] = registry.scaler.transform(input_df[numerical_cols])

            # 6. Run Credit Eligibility Assessment Prediction
            prediction = int(registry.model.predict(input_df)[0])
            
            # Get probability confidence if supported
            if hasattr(registry.model, "predict_proba"):
                probabilities = registry.model.predict_proba(input_df)[0]
                confidence = float(probabilities[prediction]) * 100
                approval_probability = float(probabilities[0]) * 100
            else:
                confidence = 100.0
                approval_probability = 100.0 if prediction == 0 else 0.0

            # 7. Evaluate Risk level boundaries into Eligibility Levels
            if approval_probability >= 80.0:
                eligibility_level = "High"
            elif approval_probability >= 60.0:
                eligibility_level = "Medium"
            else:
                eligibility_level = "Low"

            # 8. Set up final result template properties
            # verdict is mapped to "Eligible" / "Not Eligible"
            verdict = "Eligible" if prediction == 0 else "Not Eligible"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if verdict == "Eligible":
                summary_text = "Based on the provided application profile, the applicant satisfies the estimated eligibility criteria for credit card approval."
                recommendation_text = "Proceed to formal application. We recommend selecting a premium card category matching your high eligibility status."
            else:
                summary_text = "Based on the provided application profile, the applicant does not satisfy the estimated eligibility criteria for credit card approval."
                recommendation_text = "Improve credit metrics. We recommend lowering credit utilization, maintaining consistent employment, or checking with a co-applicant before re-assessing."

            return render_template('result.html',
                                   verdict=verdict,
                                   confidence=f"{confidence:.2f}%",
                                   eligibility_level=eligibility_level,
                                   prediction_date=timestamp,
                                   summary=summary_text,
                                   recommendation=recommendation_text)

        except Exception as e:
            # Handle all exceptions gracefully and render with a friendly error page warning
            print(f"[ERROR] Assessment handler failed: {str(e)}")
            return render_template('predict.html', error_alert=f"Assessment Failed: {str(e)}")

    # GET request
    return render_template('predict.html')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
