from gesture_classification.application.app import init_application

application = init_application()

# ToDo: Implement model parameter config yml & model dictionary

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        application,
        host="0.0.0.0",
        port=8000,
        log_level="debug",
    )
