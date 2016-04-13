import psycopg2

class TrackData:
    def __init__(self):
        self.conn = None
        self.cur = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect("dbname='spotifyechonest' user='owner' host='localhost' password='h4ck3r'")
            self.cur = self.conn.cursor()
        except:
            print "Unable to connect to the database."

    # retrieve the data from the database
    def retrieve_data(self, limit_total, index_of_attribute_to_classify):
        sql = """SELECT name,
            id,
            round(danceability::numeric, 8),
            round(loudness::numeric, 8),
            round(energy::numeric, 8),
            round(speechiness::numeric, 8),
            round(liveness::numeric, 8),
            round(acousticness::numeric, 8),
            round(instrumentalness::numeric, 8) FROM tracks
            ORDER BY id ASC LIMIT %s;"""
        self.cur.execute(sql, [limit_total])

        data = []
        for index, row in enumerate(self.cur.fetchall()):
            data.append([])
            for j, col in enumerate(row):
                if j == index_of_attribute_to_classify + 2:
                    if col > 0.5:
                        data[index].append(1)
                    else:
                        data[index].append(0)
                else:
                    data[index].append(col)

        return data

    def close(self):
        self.cur.close()
        self.conn.close()
