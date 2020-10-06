from django.views import View
from django.http import JsonResponse
from django.db import connection


class IndexView(View):
    @staticmethod
    def get(request):
        with connection.cursor() as cursor:
            cursor.execute("""\
            SELECT rank() over (ORDER by a.level desc) as rank, a.id, a.username, a.level, count(b.user_id) as captured\
             FROM accounts_user as a LEFT JOIN animals_post as b\
             on a.id = b.user_id\
            GROUP by a.id\
            """)

            row = cursor.fetchall()
            rank_list = []

            for element in row:
                rank_object = {'rank': element[0], 'id': element[1], 'username': element[2], 'level': element[3], 'captured': element[4]}
                rank_list.append(rank_object)

            return JsonResponse({'data': rank_list})
