from datetime import datetime

from streamlit_javascript import st_javascript

def getDateAndTimezone():
    date = datetime.today().strftime("%Y-%m-%d")
    timezone = st_javascript("""await (async () => {
        const now = new Date();

        const timezoneOffsetInMinutes = now.getTimezoneOffset();
        const offsetHours = Math.floor(Math.abs(timezoneOffsetInMinutes) / 60);
        const offsetMinutes = Math.abs(timezoneOffsetInMinutes) % 60;

        const sign = timezoneOffsetInMinutes > 0 ? "-" : "+";

        const formattedOffset = `UTC${sign}${String(offsetHours).padStart(2, '0')}:${String(offsetMinutes).padStart(2, '0')}`;

        return formattedOffset;
    })().then(returnValue => returnValue)""")

    return date, timezone