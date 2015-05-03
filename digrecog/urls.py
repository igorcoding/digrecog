from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    # Examples:
    url(r'^$', 'app.views.home', name='home'),
    url(r'^recognise/', 'app.views.recognise', name='recognise'),
    url(r'^train/', 'app.views.train', name='train'),
    url(r'^retrain/', 'app.views.retrain', name='retrain'),
    url(r'^mnist/', 'app.views.train_mnist', name='mnist'),
    # url(r'^blog/', include('blog.urls')),
    url(r'^admin/', include(admin.site.urls)),
]
