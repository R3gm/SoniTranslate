language_data = {
  "english": {
    "description": """
        ### 🎥 **Translate videos easily with SoniTranslate!** 📽️

        Upload a video or provide a video link. 📽️ **Gets the updated notebook from the official repository.: [SoniTranslate](https://github.com/R3gm/SoniTranslate)!**

        See the tab labeled `Help` for instructions on how to use it. Let's start having fun with video translation! 🚀🎉
        """,
    "tutorial" : """
        # 🔰 **Instructions for use:**

        1. 📤 **Upload a video** on the first tab or 🌐 **use a video link** on the second tab.

        2. 🌍 Choose the language in which you want to **translate the video**.

        3. 🗣️ Specify the **number of people speaking** in the video and **assign each one a text-to-speech voice** suitable for the translation language.

        4. 🚀 Press the '**Translate**' button to obtain the results.


        # 🎤 How to Use R.V.C. and R.V.C.2 Voices (Optional) 🎶

        The goal is to apply a R.V.C. to the generated TTS (Text-to-Speech) 🎙️

        1. In the `Custom Voice R.V.C.` tab, download the models you need 📥 You can use links from Hugging Face and Google Drive in formats like zip, pth, or index. You can also download complete HF space repositories, but this option is not very stable 😕

        2. Now, go to `Replace voice: TTS to R.V.C.` and check the `enable` box ✅ After this, you can choose the models you want to apply to each TTS speaker 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

        3. Adjust the F0 method that will be applied to all R.V.C. 🎛️

        4. Press `APPLY CONFIGURATION` to apply the changes you made 🔄

        5. Go back to the video translation tab and click on 'Translate' ▶️ Now, the translation will be done applying the R.V.C. 🗣️

        Tip: You can use `Test R.V.C.` to experiment and find the best TTS or configurations to apply to the R.V.C. 🧪🔍

        """,
    "tab_translate" : "Video to video translation",
    "video_source": "Choose Video Source",
    "link_info": "Example: www.youtube.com/watch?v=g_9rPvbENUw",
    "link_ph" : "URL goes here...",
    "dir_info": "Example: /usr/home/my_video.mp4",
    "dir_ph" : "Path goes here...",
    "sl_label" : "Source language",
    "sl_info": "This is the original language of the video",
    "tat_label" : "Translate audio to",
    "tat_info": "Select the target language, and make sure to select the language corresponding to the speakers of the target language to avoid errors in the process.",
    "num_speakers" : "Select how many people are speaking in the video.",
    "min_sk" : "Min speakers",
    "max_sk" : "Max speakers",
    "tts_select" : "Select the voice you want for each speaker.",
    "sk1" : "TTS Speaker 1",
    "sk2" : "TTS Speaker 2",
    "sk3" : "TTS Speaker 3",
    "sk4" : "TTS Speaker 4",
    "sk5" : "TTS Speaker 5",
    "sk6" : "TTS Speaker 6",
    "extra_setting" : "Advanced Settings",
    "acc_max_label" : "Max Audio acceleration",
    "acc_max_info" : "Maximum acceleration for translated audio segments to avoid overlapping. A value of 1.0 represents no acceleration",
    "aud_mix_label" : "Audio Mixing Method",
    "aud_mix_info" : "Mix original and translated audio files to create a customized, balanced output with two available mixing modes.",
    "vol_ori" : "Volume original audio",
    "vol_tra" : "Volume translated audio",
    "sub_type" : "Subtitle type",
    "whisper_title" : "Default configuration of Whisper.",
    "out_name_label" : "Translated file name",
    "out_name_info" : "The name of the output file",
    "preview_info" : "Preview cuts the video to only 10 seconds for testing purposes. Please deactivate it to retrieve the full video duration.",
    "edit_sub_label" : "Edit generated subtitles",
    "edit_sub_info" : "Edit generated subtitles: Allows you to run the translation in 2 steps. First with the 'GET SUBTITLES AND EDIT' button, you get the subtitles to edit them, and then with the 'TRANSLATE' button, you can generate the video",
    "button_subs" : "GET SUBTITLES AND EDIT",
    "editor_sub_label" : "Generated subtitles",
    "editor_sub_info" : "Feel free to edit the text in the generated subtitles here. You can make changes to the interface options before clicking the 'TRANSLATE' button, except for 'Source language', 'Translate audio to', and 'Max speakers', to avoid errors. Once you're finished, click the 'TRANSLATE' button.",
    "editor_sub_ph" : "First press 'GET SUBTITLES AND EDIT' to get the subtitles",
    "button_translate" : "TRANSLATE",
    "output_result_label" : "DOWNLOAD TRANSLATED VIDEO",
    "sub_ori" : "Subtitles",
    "sub_tra" : "Translated subtitles",
    "ht_token_info" : "One important step is to accept the license agreement for using Pyannote. You need to have an account on Hugging Face and accept the license to use the models: https://huggingface.co/pyannote/speaker-diarization and https://huggingface.co/pyannote/segmentation. Get your KEY TOKEN here: https://hf.co/settings/tokens",
    "ht_token_ph" : "Token goes here...",
    "cv_url_info" : "Automatically download the R.V.C. models from the URL. You can use links from HuggingFace or Drive, and you can include several links, each one separated by a comma. Example: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index",
    "tab_help" : "Help",
    "replace_title" : "Replace voice: TTS to R.V.C.",
    "sec1_title" : "### 1. To enable its use, mark it as enable.",
    "enable_replace" : "Check this to enable the use of the models.",
    "sec2_title" : "### 2. Select a voice that will be applied to each TTS of each corresponding speaker and apply the configurations.",
    "sec2_subtitle" : "Depending on how many <TTS Speaker> you will use, each one needs its respective model. Additionally, there is an auxiliary one if for some reason the speaker is not detected correctly.",
    "cv_tts1" : "Choose the voice to apply for Speaker 1.",
    "cv_tts2" : "Choose the voice to apply for Speaker 2.",
    "cv_tts3" : "Choose the voice to apply for Speaker 3.",
    "cv_tts4" : "Choose the voice to apply for Speaker 4.",
    "cv_tts5" : "Choose the voice to apply for Speaker 5.",
    "cv_tts6" : "Choose the voice to apply for Speaker 6.",
    "cv_aux" : "- Voice to apply in case a Speaker is not detected successfully.",
    "cv_button_apply" : "APPLY CONFIGURATION"
  },

  "spanish": {
    "description": """
        ### 🎥 **¡Traduce videos fácilmente con SoniTranslate!** 📽️

        Sube un video o proporciona un enlace de video. 📽️ **Obtén el cuaderno actualizado desde el repositorio oficial: [SoniTranslate](https://github.com/R3gm/SoniTranslate)!**

        Consulta la pestaña etiquetada como `Ayuda` para obtener instrucciones sobre cómo usarlo. ¡Comencemos a divertirnos con la traducción de videos! 🚀🎉
        """,
    "tutorial": """
        # 🔰 **Instrucciones de uso:**

        1. 📤 **Sube un video** en la primera pestaña o 🌐 **utiliza un enlace de video** en la segunda pestaña.

        2. 🌍 Elige el idioma en el que deseas **traducir el video**.

        3. 🗣️ Especifica el **número de personas que hablan** en el video y **asigna a cada una una voz de texto a voz** adecuada para el idioma de traducción.

        4. 🚀 Presiona el botón '**Traducir**' para obtener los resultados.


        # 🎤 Cómo usar las voces R.V.C. y R.V.C.2 (Opcional) 🎶

        El objetivo es aplicar un R.V.C. al TTS (Texto a Voz) generado 🎙️

        1. En la pestaña `Voz Personalizada R.V.C.`, descarga los modelos que necesitas 📥 Puedes utilizar enlaces de Hugging Face y Google Drive en formatos como zip, pth o index. También puedes descargar repositorios completos de espacio HF, pero esta opción no es muy estable 😕

        2. Ahora, ve a `Reemplazar voz: TTS a R.V.C.` y marca la casilla `habilitar` ✅ Después de esto, puedes elegir los modelos que deseas aplicar a cada hablante de TTS 👩‍🦰👨‍🦱👩‍🦳👨‍🦲

        3. Ajusta el método F0 que se aplicará a todos los R.V.C. 🎛️

        4. Presiona `APLICAR CONFIGURACIÓN` para aplicar los cambios que hayas realizado 🔄

        5. Vuelve a la pestaña de traducción de video y haz clic en 'Traducir' ▶️ Ahora, la traducción se realizará aplicando el R.V.C. 🗣️

        Consejo: Puedes usar `Probar R.V.C.` para experimentar y encontrar el mejor TTS o configuraciones para aplicar al R.V.C. 🧪🔍

        """,
    "tab_translate" : "Traducción de video a video",
    "video_source": "Seleccionar Fuente de Video",
    "link_info": "Ejemplo: www.youtube.com/watch?v=g_9rPvbENUw",
    "link_ph": "Ingrese la URL aquí...",
    "dir_info": "Ejemplo: /usr/home/my_video.mp4",
    "dir_ph": "Ingrese la ruta aquí...",
    "sl_label": "Idioma de origen",
    "sl_info": "Este es el idioma original del video",
    "tat_label": "Traducir audio a",
    "tat_info": "Seleccione el idioma de destino y asegúrese de seleccionar el idioma correspondiente a los hablantes del idioma objetivo para evitar errores en el proceso.",
    "num_speakers": "Seleccione cuántas personas están hablando en el video.",
    "min_sk": "Mín. de hablantes",
    "max_sk": "Máx. de hablantes",
    "tts_select": "Seleccione la voz que desea para cada hablante.",
    "sk1": "TTS Hablante 1",
    "sk2": "TTS Hablante 2",
    "sk3": "TTS Hablante 3",
    "sk4": "TTS Hablante 4",
    "sk5": "TTS Hablante 5",
    "sk6": "TTS Hablante 6",
    "extra_setting": "Configuraciones Avanzadas",
    "acc_max_label": "Máx. de Aceleración de Audio",
    "acc_max_info": "Aceleración máxima para segmentos de audio traducidos para evitar superposiciones. Un valor de 1.0 representa ninguna aceleración.",
    "aud_mix_label": "Método de Mezcla de Audio",
    "aud_mix_info": "Mezclar archivos de audio original y traducido para crear una salida personalizada y equilibrada con dos modos de mezcla disponibles.",
    "vol_ori": "Volumen audio original",
    "vol_tra": "Volumen audio traducido",
    "sub_type": "Tipo de Subtítulos",
    "whisper_title": "Configuración predeterminada de Whisper.",
    "out_name_label": "Nombre del archivo traducido",
    "out_name_info": "El nombre del archivo de salida",
    "preview_info": "La vista previa corta el video a solo 10 segundos con fines de prueba. Desactívelo para obtener la duración completa del video.",
    "edit_sub_label": "Editar subtítulos generados",
    "edit_sub_info": "Editar subtítulos generados: Permite ejecutar la traducción en 2 pasos. Primero, con el botón 'OBTENER SUBTÍTULOS Y EDITAR', obtiene los subtítulos para editarlos, y luego con el botón 'TRADUCIR', puede generar el video.",
    "button_subs": "OBTENER SUBTÍTULOS Y EDITAR",
    "editor_sub_label": "Subtítulos generados",
    "editor_sub_info": "Siéntase libre de editar el texto de los subtítulos generados aquí. Puede realizar cambios en las opciones de la interfaz antes de hacer clic en el botón 'TRADUCIR', excepto en 'Idioma de origen', 'Traducir audio a' y 'Máx. de hablantes', para evitar errores. Una vez que haya terminado, haga clic en el botón 'TRADUCIR'.",
    "editor_sub_ph": "Presione primero 'OBTENER SUBTÍTULOS Y EDITAR' para obtener los subtítulos",
    "button_translate": "TRADUCIR",
    "output_result_label": "DESCARGAR VIDEO TRADUCIDO",
    "sub_ori": "Subtítulos originales",
    "sub_tra": "Subtítulos traducidos",
    "ht_token_info": "Un paso importante es aceptar el acuerdo de licencia para usar Pyannote. Debe tener una cuenta en Hugging Face y aceptar la licencia para usar los modelos: https://huggingface.co/pyannote/speaker-diarization y https://huggingface.co/pyannote/segmentation. Obtenga su TOKEN aquí: https://hf.co/settings/tokens",
    "ht_token_ph": "Ingrese el token aquí...",
    "cv_url_info": "Descargue automáticamente los modelos R.V.C. desde la URL. Puede utilizar enlaces de HuggingFace o Drive, e incluso puede incluir varios enlaces, cada uno separado por una coma. Ejemplo: https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.pth, https://huggingface.co/sail-rvc/yoimiya-jp/blob/main/model.index",
    "tab_help": "Ayuda",
    "replace_title": "Reemplazar voz: TTS a R.V.C.",
    "sec1_title": "### 1. Para habilitar su uso, márquelo como habilitado.",
    "enable_replace": "Marque esto para habilitar el uso de los modelos.",
    "sec2_title": "### 2. Seleccione una voz que se aplicará a cada TTS de cada hablante correspondiente y aplique las configuraciones.",
    "sec2_subtitle": "Dependiendo de cuántos <TTS Hablante> vaya a usar, cada uno necesita su respectivo modelo. Además, hay uno auxiliar si por alguna razón el hablante no es detectado correctamente.",
    "cv_tts1": "Voz a aplicar al TTS Hablante 1.",
    "cv_tts2": "Voz a aplicar al TTS Hablante 2.",
    "cv_tts3": "Voz a aplicar al TTS Hablante 3.",
    "cv_tts4": "Voz a aplicar al TTS Hablante 4.",
    "cv_tts5": "Voz a aplicar al TTS Hablante 5.",
    "cv_tts6": "Voz a aplicar al TTS Hablante 6.",
    "cv_aux": "- Voz a aplicar en caso de que un hablante no sea detectado correctamente.",
    "cv_button_apply": "APLICAR CONFIGURACIÓN"
  }

}
