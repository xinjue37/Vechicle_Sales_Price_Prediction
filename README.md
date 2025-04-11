# Vechicle_Sales_Price_Prediction
* Python Version used: 3.10.x
* To install all dependency: 
```
  pip install -r requirements.txt
```
* Running whole jupyter notebook required ~ 25 minutes
* Time taken to complete this project: ~ 20h

## Directory Structure
```
📁 project_root/
├── 📂 data/
    |- 📊 DataSample.rpt
├── 📂 inference_pipeline/  # Generated during run time
├── 📂 models/              # Generated during run time
├── 📘 main.ipynb           # Main code file
├── 📄 README.md            # This file
├── 📦 requirements.txt     # Required python dependency
├── 📊 sample_input.csv     # Sample input to models
└── 🛠️ utility.py           # All utility function/class
```

## Vechicle Price Prediction Analysis Process

| Step | Description |Analysis|
|------|-------------|:--|
| **1. Initial Considerations/Thought** | Generally, to perform valuation on a car, we needs: <ul><li>Car Brand</li><li>Initial Price</li><li>Car Mileage (Odometer)</li><li>Car Age</li><li>Exterior & Interior Condition</li><li>Trim Level (BadgeDescription)</li><li>Accident History (from VIN)</li><li>Market Demand</li><li>Car Apperance</li></ul> |--|
| **2. Basic Data Understanding** | <ul><li>62,192 data points</li><li>130 features (64 float64, 60 object, 6 int64)</li><li>Features grouped into **10 categories (scroll to below for details)**</li><li>44 records where Sold_Amount > NewPrice (valid due to resale/antique value)</li></ul> |--|
| **3. Feature Removal** | Remove features that are: <ul><li><b>Redundant</b>: MakeCode (Make), FamilyCode (Model), SeriesModelYear (YearGroup)</li><li><b>Identifiers</b>: SequenceNum, DriveCode, VIN, ModelCode, EngineNum</li><li><b>Constant Value</b>: ImportFlag, NormalChargeMins, NormalChargeVoltage, TopSpeedElectricEng</li><li><b>Leakage/Cause degenerate feedback loop</b>: AvgWholesale, AvgRetail, GoodWholesale, GoodRetail, TradeMin, TradeMax, PrivateMax</li></ul> |--|
|**4. Exploratory Data Analysis (EDA)**|<ul><li>To find out some possible useful insights from the features</li><li>just perform simple EDA due to time constraints</li></ul>|This dataset contains cars from 1962 to 2017, with missing values being common and outliers frequently present due to the inclusion of luxury vehicles. The distribution of **Sold_Amount** is heavily right-skewed, with most vehicles sold below 40,000, and some high-value outliers likely representing luxury vehicles. The correlation heatmap shows a weak positive correlation between NewPrice and Sold_Amount, and a weak negative correlation between Age_Comp_Months and KM with Sold_Amount. This is expected, as Sold_Amount tends to align with the original price and decreases with increased usage and age.|
| **5. Feature Engineering** | <ul><li>Add accident history (from VIN) - **left it aside due to time constraint**</li><li>Incorporate car appearance (via images) - **left it aside due to time constraint**</li><li>Convert date fields into year, month, day, weekday</li></ul> |--|
| **6. Encoding Categorical Data** | **One-Hot Encoding with "Other" Category:** <br>* Reduces dimensionality by collapsing low-frequency categories into "Other."<br>* Pros: Prevents rare/unseen categories from breaking inference pipelines.<br>* Cons: Loss of granularity; "Other" may dilute meaningful signals.<br><br>**Target Encoding:**<br>* Maps categories to the mean target value.<br>* Risks: Data leakage if encoding is not strictly separated during cross-validation, leading to overoptimistic performance.<br><br>**Feature Hashing (n=100)**: <br>* Projects categories into a fixed-dimensional space via hash functions.<br>* Pros: Memory-efficient, handles unseen categories gracefully.<br>* Cons: Irreversible (loss of interpretability), potential hash collisions.|--|
| **7. Model Building & Training** | **Algorithm Choice**: Tree-based models (XGBoost, CatBoost) were selected for inherent handling of missing data and nonlinear relationships.<br><br>**Hyperparameter Tuning**: Limited tuning (n_iter=10, cv=3) due to resource constraints and this is just a prove of concept. |--|
| **8. Model Evaluation** | Perform slice-based evaluation by the `Make` feature to analyze model performance across car brands.  |Among **three different embedding methods**, one-hot encoding resulted in higher performance overall. <br>Among **three different tree-based ML models**, they performed similarly, with no significant variation in **R²**, **RMSE**, or **MAE**. <br>This suggests that categorical features are underutilized, implying the model relies primarily on **NewPrice**, **Age_Comp_Months**, and **KM**.|
| **9. Feature Importance Analysis** |Analyze and compare which features are most important to each model.|The consistent high importance of "NewPrice", "Age_Comp_Months" & "KM" across different models (XGBoost and CatBoost) and various categorical encoding techniques (One-hot encoding, Target encoding, and Feature hashing) strongly suggests that the original price of a vehicle, the car's age, and the car Mileage is a fundamental predictor of its current market value.|
| **10. Inference Pipeline** | <ul><li>Build a pipeline for making predictions on unseen data.</li><li>show how it handles new categorical data</li></ul>|--|

>  **CONCLUSION FROM ANALYSIS**: In conclusion, the limitations of current models in leveraging categorical features, such as vehicle descriptions, hinder optimal car price prediction. The inability of traditional encoding methods to capture semantic and hierarchical relationships within the data highlights the need for more advanced techniques, such as neural network-based entity embeddings or BERT for textual inputs. Furthermore, enriching the feature space with market trends and vehicle-specific attributes can provide additional context, ultimately leading to more accurate and reliable vehicle price predictions.

## Possible Strategic Recommendations for Business Optimization
| **Possible Enhancement**                     | **Details** |
|----------------------------------|-------------|
| **A) Enhance Categorical Feature Representation** | <ul><li>Experiment with hierarchical embeddings (e.g., brand → model → year) or NLP techniques for text descriptions.</li><li>Collect additional metadata (e.g., trim level, optional features) to reduce ambiguity.</li><li>After categorical features are well used, we can possibly move on to analyse the impact of unseen categories on the ML models.</li></ul> |
| **B) Luxury Car Segmentation**   | <ul><li>Luxury cars represent niche markets with distinct pricing dynamics. Failing to model these outliers accurately could lead to underpricing high-margin inventory.</li><li>Train separate models for luxury vs. non-luxury vehicles to address skewness and distinct pricing factors (e.g., brand prestige, limited editions).</li></ul> |
| **C) Dynamic Pricing Integration** | <ul><li>Incorporate real-time market data (e.g., fuel prices, competitor listings) to adapt to demand fluctuations.</li></ul> |
| **D) Model Robustness**          | <ul><li>Increase hyperparameter tuning iterations (`n_iter ≥ 50`) and cross-validation folds (`cv=5`) for production-grade models.</li></ul> |
| **E) Business Metrics Alignment** | <ul><li>Map RMSE/MAE to profit margins (e.g., a `1,000` prediction error equates to `X` loss per transaction).</li><li>Use slice-based evaluation to identify underperforming segments (e.g., electric vehicles) for targeted improvements.</li></ul> |


## 2. Basic Data Understanding
### Groups 130 features into 10 categories
| No  | Classification  | Attribute | Possible Description |
|-----|------------------------------------------|----------------------------------|----------------------------------------------------------------------------------------|
| 1 | Manufacturer & Model Information | Make | Manufacturer (e.g., Holden, Toyota) |
| 2 | Manufacturer & Model Information | Model  | Car model name (e.g., Commodore, RAV4)  |
| 3 | Manufacturer & Model Information | MakeCode | Manufacturer code (e.g., HOLD for Holden) - **REDUNDANT (Make)**  |
| 4 | Manufacturer & Model Information | FamilyCode | Internal code for the vehicle family - **REDUNDANT (Model)**  |
| 5 | Manufacturer & Model Information | Series | Generation or variant (e.g., VE, VR, ACA33R)  |
| 6 | Manufacturer & Model Information | SeriesModelYear | Model year of the series (e.g., MY12, MY88, Series IV)  |
| 7 | Manufacturer & Model Information | BadgeDescription  | Trim level (e.g., Omega, Executive) |
| 8 | Manufacturer & Model Information | BadgeSecondaryDescription | Secondary trim details (often empty)  |
| 9 | Manufacturer & Model Information | OptionCategory  | Vehicle type (e.g., PASS, SUV, VAN, BUS)  |
| 10  | Manufacturer & Model Information | VFactsClass | Market classification (e.g., Passenger, SUV)  |
| 11  | Manufacturer & Model Information | VFactsSegment | Size category (e.g., Large, Medium) |
| 12  | Manufacturer & Model Information | VFactsPrice | Price range (e.g., <$70K) |
| 13  | Vehicle Unique identifiers & General Info| YearGroup  | Model year (e.g., 2008) |
| 14  | Vehicle Unique identifiers & General Info| MonthGroup | Month of production (e.g., 0 = unknown) |
| 15  | Vehicle Unique identifiers & General Info| SequenceNum  | Unique identifier - **USELESS** |
| 16  | Vehicle Unique identifiers & General Info| Description  | Detailed description (e.g., VE Omega Sedan...)  |
| 17  | Vehicle Unique identifiers & General Info| CurrentRelease | Is the model current? (F = False, T = True) |
| 18  | Vehicle Unique identifiers & General Info| ImportFlag | Import status (L = locally made) - **USELESS (1 unique value)** |
| 19  | Vehicle Unique identifiers & General Info| LimitedEdition | Limited edition? (F = False, T = True)  |
| 20  | Vehicle Unique identifiers & General Info| BodyStyleDescription | Body type (e.g., Sedan, Wagon)  |
| 21  | Vehicle Unique identifiers & General Info| BodyConfigDescription  | Body configuration (mostly empty) |
| 22  | Vehicle Unique identifiers & General Info| WheelBaseConfig  | Wheelbase type  |
| 23  | Vehicle Unique identifiers & General Info| Roofline | Roof design |
| 24  | Vehicle Unique identifiers & General Info| ExtraIdentification  | Additional identifiers (empty)  |
| 25  | Vehicle Unique identifiers & General Info| DriveDescription | Drivetrain (e.g., Rear Wheel Drive) |
| 26  | Vehicle Unique identifiers & General Info| DriveCode  | Drivetrain code (e.g., RWD) - **USELESS**  |
| 27  | Vehicle Unique identifiers & General Info| ModelCode  | Internal model code - **USELESS** |
| 28  | Vehicle Unique identifiers & General Info| BuildCountryOriginDescription  | Manufacturing country (e.g., Australia, Japan, Thailand)  |
| 29  | Vehicle Unique identifiers & General Info| VIN  | Vehicle Identification Number - **USELESS** |
| 30  | Technical Specifications | GearTypeDescription  | Transmission type (e.g., Automatic, Manual, Sports Auto)  |
| 31  | Technical Specifications | GearLocationDescription  | Gear lever position (e.g., Floor, Dash) |
| 32  | Technical Specifications | GearNum  | Number of gears (1-9) |
| 33  | Technical Specifications | DoorNum  | Number of doors (2-5) |
| 34  | Technical Specifications | EngineSize | Engine displacement (cc) (659-7300) |
| 35  | Technical Specifications | EngineDescription  | Engine name (e.g., 3.6i, 13B, 800) |
| 36  | Technical Specifications | Cylinders  | Number of cylinders (2-12)  |
| 37  | Technical Specifications | FuelTypeDescription  | Fuel type (e.g., Petrol, Diesel, LPG) |
| 38  | Technical Specifications | InductionDescription | Aspiration type (e.g., Aspirated, Turbocharged) |
| 39  | Technical Specifications | CamDescription | Valve mechanism (e.g., DOHC with VVT) |
| 40  | Technical Specifications | EngineTypeDescription  | Engine type (e.g., Piston, Rotary)  |
| 41  | Technical Specifications | FuelDeliveryDescription  | Fuel injection type (e.g., Multi-Point) |
| 42  | Technical Specifications | MethodOfDeliveryDescription  | Fuel delivery method (e.g., Electronic) |
| 43  | Technical Specifications | ValvesCylinder | Valves per cylinder (2-5) |
| 44  | Technical Specifications | EngineCycleDescription | Engine cycle (e.g., 4 Stroke) |
| 45  | Technical Specifications | EngineConfigurationDescription | Engine layout (e.g., V6)  |
| 46  | Technical Specifications | EngineLocation | Engine placement (e.g., Front)  |
| 47  | Technical Specifications | EngineNum  | Engine serial number - **USELESS**  |
| 48  | Technical Specifications | FrontTyreSize  | Front tire dimensions (e.g., 225/60 R16)  |
| 49  | Technical Specifications | RearTyreSize | Rear tire dimensions  |
| 50  | Technical Specifications | FrontRimDesc | Front rim size (e.g., 16x7.0) |
| 51  | Technical Specifications | RearRimDesc  | Rear rim size |
| 52  | Dimensions & Weight  | WheelBase  | Distance between axles (2–4332 mm)  |
| 53  | Dimensions & Weight  | Height | Vehicle height (mm) |
| 54  | Dimensions & Weight  | Length | Vehicle length (mm) |
| 55  | Dimensions & Weight  | Width  | Vehicle width (mm)  |
| 56  | Dimensions & Weight  | KerbWeight | Weight with fluids (kg) |
| 57  | Dimensions & Weight  | TareMass | Empty weight (kg) |
| 58  | Dimensions & Weight  | PayLoad  | Max load capacity (260–2701 kg) |
| 59  | Dimensions & Weight  | SeatCapacity | Number of seats (2–15)  |
| 60  | Dimensions & Weight  | FuelCapacity | Fuel tank size (32–180 L) |
| 61  | Performance Metrics  | Power  | Engine power (kW) (15–640)  |
| 62  | Performance Metrics  | PowerRPMFrom | RPM where peak power starts |
| 63  | Performance Metrics  | PowerRPMTo | RPM where peak power ends |
| 64  | Performance Metrics  | Torque | Torque (Nm) (24–1400) |
| 65  | Performance Metrics  | TorqueRPMFrom  | RPM where peak torque starts  |
| 66  | Performance Metrics  | TorqueRPMTo  | RPM where peak torque ends  |
| 67  | Performance Metrics  | Acceleration  | 0-100 km/h time (mostly empty)  |
| 68  | Performance Metrics  | TowingBrakes | Towing capacity with brakes (kg)  |
| 69  | Performance Metrics  | TowingNoBrakes | Towing capacity without brakes (kg) |
| 70  | Performance Metrics  | TopSpeedElectricEng | Top speed on electric power - **USELESS - exactly 1 unique value**.  |
| 71  | Fuel & Emissions | RonRating  | Octane rating (e.g., 91, 95)  |
| 72  | Fuel & Emissions | FuelUrban  | Urban fuel consumption  |
| 73  | Fuel & Emissions | FuelExtraurban | Highway fuel consumption  |
| 74  | Fuel & Emissions | FuelCombined | Combined fuel consumption |
| 75  | Fuel & Emissions | CO2Combined  | Combined CO2 emissions  |
| 76  | Fuel & Emissions | CO2Urban | Urban CO2 emissions |
| 77  | Fuel & Emissions | CO2ExtraUrban  | Highway CO2 emissions |
| 78  | Fuel & Emissions | EmissionStandard | Compliance standard  |
| 79  | Fuel & Emissions | MaxEthanolBlend | Ethanol compatibility  |
| 80  | Fuel & Emissions | GreenhouseRating | Greenhouse score (1–10) |
| 81  | Fuel & Emissions | AirpollutionRating | Air pollution score (1–10)  |
| 82  | Fuel & Emissions | OverallGreenStarRating | Environmental star rating (1–5) |
| 83  | Safety & Compliance  | AncapRating  | ANCAP safety rating (0–5 stars) |
| 84  | Safety & Compliance  | GrossCombinationMass | Maximum combined weight |
| 85  | Safety & Compliance  | GrossVehicleMass | Gross vehicle mass  |
| 86  | Safety & Compliance  | IsPPlateApproved | Is P-plate approved (T/F) |
| 87  | Sales & Pricing  | AverageKM  | Average odometer reading  |
| 88  | Sales & Pricing  | GoodKM | Low-KM threshold  |
| 89  | Sales & Pricing  | AvgWholesale | Average wholesale price **Cant use it, make result degenerate feedback loop**.  |
| 90  | Sales & Pricing  | AvgRetail  | Average retail price **Cant use it, make result degenerate feedback loop**.  |
| 91  | Sales & Pricing  | GoodWholesale | Wholesale price for low-KM cars **Cant use it, make result degenerate feedback loop**. |
| 92  | Sales & Pricing  | GoodRetail  | Retail price for low-KM cars **Cant use it, make result degenerate feedback loop**.  |
| 93  | Sales & Pricing  | TradeMin  | Minimum trade-in value **Cant use it, make result degenerate feedback loop**. |
| 94  | Sales & Pricing  | TradeMax  | Maximum trade-in value **Cant use it, make result degenerate feedback loop**. |
| 95  | Sales & Pricing  | PrivateMax | Maximum private sale price **Cant use it, make result degenerate feedback loop**. |
| 96  | Sales & Pricing  | NewPrice | Original new price **This features is very important!, cannot simply impute the nan**.  |
| 97  | Sales & Pricing  | Colour | Vehicle color |
| 98  | Sales & Pricing  | Branch | Dealership location |
| 99  | Sales & Pricing  | SaleCategory | Type of sale (e.g., Auction). |
| 100 | Sales & Pricing  | Sold_Date  | Date sold **Also very important features, convert it into year,month,day**.  |
| 101 | Sales & Pricing  | Compliance_Date  | Date vehicle complied with regulations  |
| 102 | Sales & Pricing  | Age_Comp_Months  | Age of vehicle in months at compliance  |
| 103 | Sales & Pricing  | KM | Odometer reading at sale  |
| 104 | Sales & Pricing  | **Sold_Amount**  | Final sold price  |
| 105 | Service & Warranty | WarrantyCustAssist | Roadside assistance coverage duration |
| 106 | Service & Warranty | FreeScheduledService  | Free services |
| 107 | Service & Warranty | WarrantyYears  | Warranty duration in years  |
| 108 | Service & Warranty | WarrantyKM | Warranty kilometers |
| 109 | Service & Warranty | FirstServiceKM | Distance for first service  |
| 110 | Service & Warranty | FirstServiceMonths | Time for first service (months) |
| 111 | Service & Warranty | RegServiceMonths | Regular service interval in months  |
| 112 | Electric/Hybrid Vehicle Specifications | AltEngEngineType | Alternate engine type |
| 113 | Electric/Hybrid Vehicle Specifications | AltEngBatteryType | Battery type  |
| 114 | Electric/Hybrid Vehicle Specifications | AltEngCurrentType | Current type (AC/DC)  |
| 115 | Electric/Hybrid Vehicle Specifications | AltEngAmpHours  | Battery capacity (Ah) |
| 116 | Electric/Hybrid Vehicle Specifications | AltEngVolts | Battery voltage |
| 117 | Electric/Hybrid Vehicle Specifications | AltEngChargingMethod  | Charging method |
| 118 | Electric/Hybrid Vehicle Specifications | AltEngPower | Electric motor power  |
| 119 | Electric/Hybrid Vehicle Specifications | AltEngPowerFrom | Power RPM start |
| 120 | Electric/Hybrid Vehicle Specifications | AltEngPowerTo | Power RPM end |
| 121 | Electric/Hybrid Vehicle Specifications | AltEngTorque  | Torque (Nm) |
| 122 | Electric/Hybrid Vehicle Specifications | AltEngTorqueFrom  | Torque RPM start  |
| 123 | Electric/Hybrid Vehicle Specifications | AltEngTorqueTo  | Torque RPM end  |
| 124 | Electric/Hybrid Vehicle Specifications | AltEngDrive | Electric drivetrain |
| 125 | Electric/Hybrid Vehicle Specifications | NormalChargeMins | Standard charging time - **USELSES (exactly 1 unique value)**.|
| 126 | Electric/Hybrid Vehicle Specifications | QuickChargeMins | Fast charging time (minutes)  |
| 127 | Electric/Hybrid Vehicle Specifications | NormalChargeVoltage | Standard charging voltage - **USELSES (exactly 1 unique value)**. |
| 128 | Electric/Hybrid Vehicle Specifications | QuickChargeVoltage  | Fast charging voltage |
| 129 | Electric/Hybrid Vehicle Specifications | KMRangeElectricEng  | Electric-only range in km |
| 130 | Electric/Hybrid Vehicle Specifications | ElectricEngineLocation  | Electric motor placement  |

