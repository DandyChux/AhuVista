[![LGPL License](https://img.shields.io/badge/License-LGPL-green.svg)](https://choosealicense.com/licenses/lgpl-3.0/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/License-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)
[![Solid.js](https://img.shields.io/badge/Frontend-Solid.js-blue.svg)](http://www.solidjs.com)
[![Rust](https://img.shields.io/badge/Backend-Rust-orange.svg)](http://www.rust-lang.org)
[![Rust](https://img.shields.io/badge/Machine_Learning-Rust-orange.svg)](http://www.rust-lang.org)


# ÀhuVista

ÀhuVista is a high-performance application designed to enhance patient care by predicting maternal outcomes using cutting-edge technologies. Built with a stack of Rust & React, the application leverages the computational efficiency and safety of Rust for data processing and machine learning, and backend API development, and the responsiveness and versalitity of Solid.js for a dynamic user interface.

### Objective

The primary objective of ÀhuVista is to provide healthcare professionals with a reliable tool for predicting patient outcomes based on clinical data, while also empowering patients with a platform to interact with their health data and gain insights into potential health outcomes for informed discussions with their medical professionals. This predictive capability aims to support clinical decision-making, improve patient management strategies, and enhance the overall efficiency of healthcare services.

## Features

- **Data Processing:** Utilizes Rust for its superior performance in handling complex data processing and machine learning tasks, ensuring quick and accurate predictions.
- **API and Business Logic:** Developed in Rust, the backend supports high performance and memory safety for efficient handling of API requests, facilitating seamless data flow and interaction.
- **User Interface:** Implemented with Solid.js, the frontend offers an engaging and intuitive user experience, making it easy for medical staff to input data and interpret predictive results, while also providing patietns with accessible tools to view and understand their health data.
- **Scalability and Security:** Designed to scale seamlessly with increasing data loads while ensuring data security and compliance with healthcare regulations such as HIPAA.

### Use Cases

- Assisting doctors in assessing patient risks and prognoses.
- Helping hospitals manage patient care more effectively.
- Supporting medical research by providing data-driven insights into patient outcomes.
- Enabling patients to access and understand their health data and potential outcomes to facilitate meaningful conversations with healthcare providers.

This project not only bridges the gap between technology and healthcare but also aims to be a pivotal tool in advancing medical research and patient care.


## Tech Stack

**Client:** Solid.js, TailwindCSS

**Server:** Rust, Tokio

**Machine Learning:** Rust


## API Reference

#### User signup

```http
  POST /auth/signup
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `email` | `string` | **Required**. User's email address |
| `password` | `string` | **Required**. User's password |
| `user_type` | `UserType` | **Required**. Type of user account |

#### User signin

```http
  POST /auth/signin
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `email`      | `string` | **Required**. User's email address |
| `password`      | `string` | **Required**. User's password |

#### add(num1, num2)

Takes two numbers and returns the sum.
