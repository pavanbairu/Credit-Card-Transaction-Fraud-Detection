# Credit-Card-Transaction-Fraud-Detection

## Problem Statement
Credit card fraud is a significant financial threat. This project focuses on identifying fraudulent transactions using machine learning techniques on a simulated dataset.

## Dataset Overview
- **Source**: [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection/data) (Simulated transactions using Sparkov Data Generation)
- **Duration**: January 1, 2019 – December 31, 2020
- **Customers**: 1,000
- **Merchants**: 800
- **Transactions**: Includes both legitimate and fraudulent transactions

## Columns Description
| Column Name               | Description                                             |
|---------------------------|---------------------------------------------------------|
| `index`                   | Unique row ID                                           |
| `trans_date_trans_time`   | Transaction timestamp                                   |
| `cc_num`                  | Customer’s credit card number                           |
| `merchant`                | Merchant name                                           |
| `category`                | Merchant category                                       |
| `amt`                     | Transaction amount                                      |
| `first`                   | Cardholder’s first name                                 |
| `last`                    | Cardholder’s last name                                  |
| `gender`                  | Cardholder’s gender                                     |
| `street`                  | Cardholder’s street address                             |
| `city`                    | Cardholder’s city                                       |
| `state`                   | Cardholder’s state                                      |
| `zip`                     | Cardholder’s ZIP code                                   |
| `lat`                     | Cardholder’s latitude                                   |
| `long`                    | Cardholder’s longitude                                  |
| `city_pop`                | Population of the cardholder’s city                     |
| `job`                     | Cardholder’s job title                                  |
| `dob`                     | Cardholder’s date of birth                              |
| `trans_num`               | Unique transaction ID                                   |
| `unix_time`               | Transaction timestamp in UNIX format                    |
| `merch_lat`               | Merchant location latitude                              |
| `merch_long`              | Merchant location longitude                             |
| `is_fraud`                | Fraud flag (Target variable: 0 = Legitimate, 1 = Fraud) |

## How to run?
### download or clone project to local machine and run the following commands

```bash
conda create -n venv python=3.11 -y
```

```bash
conda activate venv
```

```bash
pip install -r requirements.txt
```

```bash
python app.py
```


## Workflow

1. constant
2. config_entity
3. artifact_entity
4. component
5. pipeline
6. app.py / demo.py


### Export the  environment variable
```bash
export AWS_REGION=<AWS_REGION>

export AWS_ACCESS_KEY_ID=<AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<AWS_SECRET_ACCESS_KEY>
```



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

    3. AmazonS3FullAccess

	
## 3. Create ECR repo to store/save docker image
    - the URI: <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/creditcard-fraud-detection

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting -> actions -> runner -> new self hosted runner -> choose os -> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_REGION
   - ECR_REPOPOSITORY_NAME
   - AWS_ECR_LOGIN_URI
