# backend/models/patient.py

class Patient:
    def __init__(
        self,
        userID: int = None,
        name: str = "",
        age: int = 0,
        gender: str = "",
        pregnancies: int = 0,
        glucose: float = 0.0,
        bloodPressure: float = 0.0,
        skinThickness: float = 0.0,
        insulin: float = 0.0,
        BMI: float = 0.0,
        diabetesPedigreeFunction: float = 0.0
    ):
        # Attributes (as per class diagram)
        self.userID = userID
        self.name = name
        self.age = age
        self.gender = gender
        self.pregnancies = pregnancies
        self.glucose = glucose
        self.bloodPressure = bloodPressure
        self.skinThickness = skinThickness
        self.insulin = insulin
        self.BMI = BMI
        self.diabetesPedigreeFunction = diabetesPedigreeFunction

    # +inputPatientData() : void
    def inputPatientData(self, data: dict):
        """
        Accepts patient data from UI / API and assigns values
        """
        self.userID = data.get("userID")
        self.name = data.get("name")
        self.age = data.get("age")
        self.gender = data.get("gender")
        self.pregnancies = data.get("pregnancies")
        self.glucose = data.get("glucose")
        self.bloodPressure = data.get("bloodPressure")
        self.skinThickness = data.get("skinThickness")
        self.insulin = data.get("insulin")
        self.BMI = data.get("BMI")
        self.diabetesPedigreeFunction = data.get("diabetesPedigreeFunction")

    # +displayPatientData() : void
    def displayPatientData(self):
        """
        Displays patient details (for debugging / report)
        """
        return {
            "userID": self.userID,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "pregnancies": self.pregnancies,
            "glucose": self.glucose,
            "bloodPressure": self.bloodPressure,
            "skinThickness": self.skinThickness,
            "insulin": self.insulin,
            "BMI": self.BMI,
            "diabetesPedigreeFunction": self.diabetesPedigreeFunction
        }
