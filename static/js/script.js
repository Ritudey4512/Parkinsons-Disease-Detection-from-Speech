const form =
    document.getElementById("predictionForm");

const resultCard =
    document.getElementById("resultCard");

const resultTitle =
    document.getElementById("resultTitle");

const probability =
    document.getElementById("probability");

const resultMessage =
    document.getElementById("resultMessage");

const resultAdvice =
    document.getElementById("resultAdvice");

const sampleBtn =
    document.getElementById("sampleBtn");

sampleBtn.addEventListener(
    "click",
    async () => {

        sampleBtn.disabled = true;

        sampleBtn.textContent =
            "Loading Sample...";

        resultCard.style.display =
            "block";

        resultTitle.textContent =
            "Loading...";

        probability.textContent =
            "—";

        resultMessage.textContent =
            "Selecting a random sample from the dataset.";

        resultAdvice.textContent =
            "Please wait.";

        try {

            const response =
                await fetch(
                    "/sample"
                );

            const result =
                await response.json();

            if (
                !response.ok ||
                result.error
            ) {

                throw new Error(
                    result.error ||
                    "Could not load sample."
                );
            }

            const sample =
                result.sample;

            Object.entries(sample).forEach(
                ([name, value]) => {

                    const input =
                        form.querySelector(
                            `[name="${CSS.escape(name)}"]`
                        );

                    if (input) {

                        input.value =
                            value;
                    }
                }
            );

            resultTitle.textContent =
                `Sample ${result.sample_number} Loaded`;


            probability.textContent =
                "—";


            resultMessage.textContent =
                `A random sample was selected `
                + `from ${result.total_samples} `
                + `dataset records.`;


            resultAdvice.textContent =
                "Click Predict Parkinson's "
                + "to run the model.";

        } catch (error) {

            resultTitle.textContent =
                "Sample Loading Error";

            probability.textContent =
                "—";

            resultMessage.textContent =
                error.message;

            resultAdvice.textContent =
                "Please try again.";

        } finally {

            sampleBtn.disabled =
                false;

            sampleBtn.textContent =
                "Load Different Sample";
        }
    }
);

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        const data = {};

        const formData =
            new FormData(form);


        formData.forEach(
            (value, key) => {

                data[key] =
                    Number(value);
            }
        );

        resultCard.style.display =
            "block";

        resultTitle.textContent =
            "Analyzing...";

        probability.textContent =
            "…";

        resultMessage.textContent =
            "The KNN model is processing "
            + "the 22 voice features.";

        resultAdvice.textContent =
            "Please wait.";


        try {

            const response =
                await fetch(
                    "/predict",
                    {

                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(data)
                    }
                );

            const result =
                await response.json();

            if (
                !response.ok ||
                result.error
            ) {

                throw new Error(
                    result.error ||
                    "Prediction failed."
                );
            }

            resultTitle.textContent =
                result.result;


            probability.textContent =
                `${result.probability}%`;


            resultMessage.textContent =
                result.message;

            if (
                result.prediction === 1
            ) {

                resultAdvice.textContent =
                    "A Parkinson's-related pattern "
                    + "was detected. This is not "
                    + "a medical diagnosis.";

            } else {

                resultAdvice.textContent =
                    "No Parkinson's-related pattern "
                    + "was detected by this model. "
                    + "This is not a medical diagnosis.";
            }


        } catch (error) {

            resultTitle.textContent =
                "Prediction Error";

            probability.textContent =
                "—";

            resultMessage.textContent =
                error.message;

            resultAdvice.textContent =
                "Please check your inputs "
                + "and try again.";
        }
    }
);