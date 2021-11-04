from django.shortcuts import render

# Create your views here.
from mysite.forms import ModelForm
from mysite.models import Predictions
import model


def test(request):
    if request.method == 'POST':
        form = ModelForm(request.POST)

        if form.is_valid():
            context = form.cleaned_data['context']

            load_model_a, load_model_b = model.predict(context)
            # print(load_model)
            Predictions.objects.create(context=context)

            return render(request, 'index.html', {'form':form,
                                                  'load_model_a': load_model_a,
                                                  'load_model_b': load_model_b})

    else:
        form = ModelForm()

    return render(request, 'index.html', {'form':form})