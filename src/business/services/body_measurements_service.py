from datetime import datetime
import numpy as np
from cv2 import imdecode, cvtColor

from fastapi import Depends, UploadFile
from persistence.repositories.repositories import BodyMeasurementsRepository
from business.handlers.service_exception import ServiceException
from business.machine_learning.measurement_predictive_model import MeasurementPredictiveModel
from domain.entities.body_measurements import BodyMeasurements
from domain.entities.measurement import Measurement
from business.mapper.mapper import Mapper
from domain.models.measurements.body_measurements_read import BodyMeasurementsRead

"""import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose"""

IMREAD_COLOR = 1
COLOR_BGR2RGB = 4

class BodyMeasurementsService():
    def __init__(
            self,
            repository: BodyMeasurementsRepository = Depends(BodyMeasurementsRepository),
            predictive_model: MeasurementPredictiveModel = Depends(MeasurementPredictiveModel),
            mapper: Mapper = Depends(Mapper)) -> None:
        self.repository = repository
        self.predictive_model = predictive_model
        self.mapper = mapper

    async def get_all(self) -> list:
        result = await self.repository.get_all()
        return result

    async def get_last_measurements(self, client_id: int) -> BodyMeasurementsRead:
        result = await self.repository.get_last_measurements(client_id)
        if result is None: return None
        result_read = self.mapper.map_to_body_measurements_read(result)
        return result_read

    async def take_measurements(
            self,
            image_frontal_file: UploadFile,
            image_lateral_file: UploadFile,
            height: float,
            client_id: int) -> BodyMeasurementsRead:
        content_frontal = await image_frontal_file.read()
        bytes_as_np_array_1 = np.frombuffer(content_frontal, dtype=np.uint8)
        image_frontal = imdecode(bytes_as_np_array_1, IMREAD_COLOR)

        content_lateral = await image_lateral_file.read()
        bytes_as_np_array_2 = np.frombuffer(content_lateral, dtype=np.uint8)
        image_lateral = imdecode(bytes_as_np_array_2, IMREAD_COLOR)

        """with  mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cvtColor(image_frontal, COLOR_BGR2RGB))
            if not results.detections:
                raise ServiceException("Las imágenes tomadas no pertenecen a una persona", 10020)"""

        """with  mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB))
            if not results.detections:
                raise ServiceException("No hay personas", 10005)"""

        arr_measurements = self.predictive_model.predict(
            image_frontal, image_lateral, height)

        entity = BodyMeasurements(
            client_id=client_id,
            measurement_date=datetime.now(),
            measurements=[
                Measurement(name_measurement='altura',
                            value=height, acronym='im', units='cm')
            ]
        )

        inserted_id = await self.repository.insert(entity)
        entity.id = inserted_id

        model = self.mapper.map_to_body_measurements_read(entity)
        return model
